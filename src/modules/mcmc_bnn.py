from collections import defaultdict
import itertools
from operator import itemgetter
from collections import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.nn as pynn
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer.reparam.strategies import AutoReparam
import pyro.distributions as dist
import pyro.infer.autoguide.initialization as ag_init

from functools import partial

from . import util


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


class NormalPrior():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.dist = dist.Normal(loc, scale)

    def sample(self, shape):
        return self.dist.sample(shape)


class BayesianLayer(pynn.PyroModule):
    def __init__(self, in_features, out_features, prior, layer_num, device="cpu", name=""):
        super().__init__(name)
        self.device = device
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.name = name
        self.layer_num = layer_num

        self.loc_w = torch.full_like(
            self.linear.weight, prior.loc, device=self.device)
        self.scale_w = torch.full_like(
            self.linear.weight, prior.scale, device=self.device)
        self.loc_b = torch.full_like(
            self.linear.bias, prior.loc, device=self.device)
        self.scale_b = torch.full_like(
            self.linear.bias, prior.scale, device=self.device)

        self.linear.weight = PyroSample(dist.Normal(
            self.loc_w, self.scale_w).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(
            self.loc_b, self.scale_b).to_event(1))

    def forward(self, x):
        return self.linear(x)


class BayesianNN(PyroModule):
    def __init__(self, in_features, out_features, prior, hidden_features=[], device="cpu", name=""):
        super().__init__(name)
        self.device = device
        self.prior = prior
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        if hidden_features == []:
            self._modules["fc0"] = BayesianLayer(
                in_features, out_features, self.prior, layer_num=0, device=self.device)
        else:
            self._modules["fc0"] = BayesianLayer(
                in_features, hidden_features[0], self.prior, layer_num=0, device=self.device)
            self._modules["act0"] = nn.ReLU()
            for i in range(len(hidden_features)-1):
                self._modules["fc"+str(i+1)] = BayesianLayer(hidden_features[i],
                                                             hidden_features[i+1], self.prior, layer_num=i+1, device=self.device)
                self._modules["act"+str(i+1)] = nn.ReLU()
            self._modules["fc"+str(len(hidden_features))] = BayesianLayer(
                hidden_features[-1], out_features, self.prior, layer_num=len(hidden_features), device=self.device)

    def forward(self, x):
        out = x
        for name, module in self._modules.items():
            if "fc" in name:
                out = module(out)
            elif "act" in name:
                out = module(out)

        return out


class GaussianLikelihood(PyroModule):
    def __init__(self, scale, dataset_size, device="cpu", data_name="obs", name=""):
        super().__init__(name)
        self.device = device
        self.data_name = data_name

        if isinstance(scale, (dist.Distribution)):
            scale = PyroSample(prior=scale)
        elif not isinstance(scale, (torch.Tensor)):
            scale = torch.tensor(scale, device=self.device)
            pass
        self._scale = scale
        self.dataset_size = dataset_size

    def forward(self, preds, obs=None):
        pred_dist = dist.Normal(preds, self._scale).to_event(1)

        dataset_size = self.dataset_size if self.dataset_size is not None else len(
            preds)
        with pyro.plate(self.data_name+"_plate", subsample=preds, size=dataset_size):
            return pyro.sample(self.data_name, pred_dist, obs=obs)

    @pynn.pyro_method
    def aggregate_predictions(self, predictions, dim=0):
        """Aggregates multiple predictions for the same data by averaging them. Predictive variance is the variance
         of the predictions plus the known variance term."""
        if isinstance(predictions, tuple):
            loc, lik_scale = predictions[0].mean(dim), predictions[1].mean(dim)
            scale = predictions[0].var(dim).add(lik_scale ** 2).sqrt()
            return loc, scale
        else:
            loc = predictions.mean(dim)
            scale = predictions.var(dim).add(self._scale ** 2).sqrt()
            return loc, scale


class MCMC_BNN(PyroModule):
    def __init__(self, model, likelihood, kernel_builder, device="cpu"):
        super().__init__()
        self.device = device
        self.model = model
        self.likelihood = likelihood
        self.kernel = kernel_builder(self._model)
        self._mcmc = None

    def _model(self, x, obs=None):
        out = self.model(*_as_tuple(x))
        self.likelihood(out, obs=obs)
        return out

    def fit(self, data_loader, num_samples, num_warmup, num_chains, batch_data=False, device=None):
        if batch_data:
            input_data, observation_data = data_loader
        else:
            input_data_lists = defaultdict(list)
            observation_data_list = []
            for in_data, obs_data in iter(data_loader):
                for i, data in enumerate(_as_tuple(in_data)):
                    input_data_lists[i].append(data.to(device))
                observation_data_list.append(obs_data.to(device))
            input_data = tuple(
                torch.cat(input_data_lists[i]) for i in range(len(input_data_lists)))
            observation_data = torch.cat(observation_data_list)

        self._mcmc = MCMC(self.kernel, num_samples=num_samples,
                          warmup_steps=num_warmup, num_chains=num_chains)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, *input_data, num_predictions=1, aggregate=True):
        if self._mcmc is None:
            raise RuntimeError(
                "Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        scales = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i]
                           for name, sample in weight_samples.items()}
                preds.append(poutine.condition(
                    self._model, weights)(*input_data))
                # sample scale from distribution
                scales.append(poutine.condition(
                    lambda: self.likelihood._scale, weights)())
        predictions = torch.stack(preds)
        scales = torch.stack(scales)
        return self.likelihood.aggregate_predictions((predictions, scales)) if aggregate else predictions
