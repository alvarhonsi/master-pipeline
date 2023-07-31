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


def get_net(in_features, out_features, hidden_features=[], activation=nn.ReLU()):
    if len(hidden_features) == 0:
        return nn.Sequential(
            nn.Linear(in_features, out_features)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_features, hidden_features[0]),
            activation,
            *[nn.Sequential(
                nn.Linear(hidden_features[i], hidden_features[i+1]),
                activation
            ) for i in range(len(hidden_features)-1)],
            nn.Linear(hidden_features[-1], out_features)
        )


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
        self._scale = scale
        self.dataset_size = dataset_size

    @pynn.pyro_method
    def get_scale(self):
        return self._scale

    def forward(self, preds, obs=None):
        pred_dist = dist.Normal(preds, self.get_scale()).to_event(1)

        dataset_size = self.dataset_size if self.dataset_size is not None else len(
            preds)
        with pyro.plate(self.data_name+"_plate", subsample=preds, size=dataset_size):
            return pyro.sample(self.data_name, pred_dist, obs=obs)

    @pynn.pyro_method
    def aggregate_predictions(self, predictions, dim=0):
        if isinstance(predictions, tuple):
            loc, lik_scale = predictions[0].mean(dim), predictions[1].mean(dim)
            scale = predictions[0].var(dim).add(lik_scale ** 2).sqrt()
            return loc, scale
        else:
            loc = predictions.mean(dim)
            scale = predictions.var(dim).add(self.get_scale() ** 2).sqrt()
            return loc, scale
