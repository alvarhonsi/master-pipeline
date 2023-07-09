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


class LR_Linear(dist.Distribution):
    def __init__(self, x, w_loc, w_scale, b_loc, b_scale):
        self.x = x
        self.w_loc = w_loc
        self.w_scale = w_scale
        self.b_loc = b_loc
        self.b_scale = b_scale

    def sample(self):
        _w_mu = torch.matmul(self.x, self.w_loc.t())
        _w_var = torch.matmul(self.x.pow(2), self.w_scale.t().pow(2))
        _w_std = torch.sqrt(1e-06 + F.softplus(_w_var))

        w = dist.Normal(_w_mu, _w_std).rsample()
        b = dist.Normal(self.b_loc, self.b_scale).rsample()

        return w + b

    def log_prob(self, value):
        _w_mu = torch.matmul(self.x, self.w_loc.t())
        _w_var = torch.matmul(self.x.pow(2), self.w_scale.t().pow(2))
        _w_std = torch.sqrt(1e-06 + F.softplus(_w_var))

        w = dist.Normal(_w_mu, _w_std).log_prob(value)
        b = dist.Normal(self.b_loc, self.b_scale).log_prob(value)

        return w + b


class NormalPrior():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.dist = dist.Normal(loc, scale)

    def sample(self, shape):
        return self.dist.sample(shape)


class LRLayer(pynn.PyroModule):
    def __init__(self, in_features, out_features, prior, device="cpu"):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

        self.loc_w = torch.full(
            (out_features, in_features), prior.loc, device=self.device, dtype=torch.float)
        self.scale_w = torch.full(
            (out_features, in_features), prior.scale, device=self.device, dtype=torch.float)
        self.loc_b = torch.full(
            (out_features,), prior.loc, device=self.device, dtype=torch.float)
        self.scale_b = torch.full(
            (out_features,), prior.scale, device=self.device, dtype=torch.float)

    def forward(self, x):
        lr_lin = LR_Linear(x, self.loc_w, self.scale_w,
                           self.loc_b, self.scale_b)
        out = pyro.sample("act", lr_lin)
        return out


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

    @pynn.pyro_method
    def _forward_lr(self, x):
        lr_lin = LR_Linear(x, self.loc_w, self.scale_w,
                           self.loc_b, self.scale_b)
        out = pyro.sample(f"act{self.layer_num}", lr_lin)
        return out

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

    def _forward_lr(self, x):
        out = x
        for name, module in self._modules.items():
            if "fc" in name:
                num = int(name[2:])
                out = module._forward_lr(out)
            elif "act" in name:
                out = module(out)

        return out

    def _forward(self, x):
        out = x
        for name, module in self._modules.items():
            if "fc" in name:
                out = module(out)
            elif "act" in name:
                out = module(out)

        return out

    def forward(self, x, y=None, lr=False):
        out = self._forward_lr(x) if lr else self._forward(x)
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


class BNNNormalGuide(PyroModule):
    def __init__(self, model, device="cpu", name="", init_fn=ag_init.init_to_median, init_scale=0.1):
        super().__init__(name)
        self.device = device
        self.model = model

        if isinstance(self.model, BayesianNN):
            dummy_x = torch.zeros(
                1, self.model.in_features, device=self.device)

            self.lr_sample_sites = pyro.poutine.trace(
                partial(self.model, lr=True)).get_trace(dummy_x).nodes.values()
        elif isinstance(self.model, GaussianLikelihood):
            dummy_x = torch.zeros(1, 1, device=self.device)

        self.sample_sites = pyro.poutine.trace(
            self.model).get_trace(dummy_x).nodes.values()

        for site in self.sample_sites:
            if "weight" in site["name"]:
                site_shape = site["value"].shape
                print(site_shape)
                loc = init_fn(site)
                # set self attr of pyro param
                attname = self.getattname(site["name"])
                setattr(self, attname+"_loc", PyroParam(
                    loc))
                setattr(self, attname+"_scale", PyroParam(
                    torch.full(site_shape, init_scale), constraint=dist.constraints.positive))
            elif "bias" in site["name"]:
                std = site["fn"].stddev.detach().shape
                loc = init_fn(site)
                # set self attr of pyro param
                attname = self.getattname(site["name"])
                setattr(self, attname+"_loc", PyroParam(
                    loc))
                setattr(self, attname+"_scale", PyroParam(
                    torch.full(std, init_scale), constraint=dist.constraints.positive))

    def getattname(self, name):
        return name.replace(".", "_")

    @pynn.pyro_method
    def _sample_lr(self, x):
        out = x
        sites = {}
        for site in self.lr_sample_sites:
            if "activation" in site["name"]:
                num = int(site["name"][10:])
                # attr names are hard coded, not ideal, but works with current implementation
                w_loc = getattr(self, f"model_fc{num}_linear_weight_loc")
                w_scale = getattr(self, f"model_fc{num}_linear_weight_scale")
                b_loc = getattr(self, f"model_fc{num}_linear_bias_loc")
                b_scale = getattr(self, f"model_fc{num}_linear_bias_scale")
                print("b_loc", b_loc)
                print("b_scale", b_scale)

                out = pyro.sample(
                    site["name"], partial(lr_linear, out, w_loc, w_scale, b_loc, b_scale))
                sites[site["name"]] = out

        return sites

    @pynn.pyro_method
    def _sample(self, x):
        sites = {}
        for site in self.sample_sites:
            if "weight" in site["name"]:
                attname = self.getattname(site["name"])
                loc = getattr(self, attname+"_loc")
                scale = getattr(self, attname+"_scale")
                sites[site["name"]] = pyro.sample(
                    site["name"], dist.Normal(loc, scale).to_event(2))
            elif "bias" in site["name"]:
                attname = self.getattname(site["name"])
                loc = getattr(self, attname+"_loc")
                scale = getattr(self, attname+"_scale")
                sites[site["name"]] = pyro.sample(
                    site["name"], dist.Normal(loc, scale).to_event(1))

        return sites

    def forward(self, x, y=None, lr=False):
        sites = self._sample_lr(x) if lr else self._sample(x)

        return sites


class SVI_BNN():
    def __init__(self, model, guide, likelihood, likelihood_guide=None, device="cpu"):
        super().__init__()
        self.device = device
        self.model = model
        self.guide = guide
        self.likelihood = likelihood
        self.likelihood_guide = likelihood_guide

    @poutine.reparam(config=AutoReparam())
    def _model(self, x, obs=None):
        out = self.model(*_as_tuple(x))
        return out

    @poutine.reparam(config=AutoReparam())
    def _guide(self, x, obs=None):
        sites = self.guide(*_as_tuple(x))
        return sites

    def fit(self, data_loader, optim, num_epochs, callback=None, num_particles=1, device=None):
        old_training_state = self.model.training
        self.model.train(True)

        loss = TraceMeanField_ELBO(num_particles)
        svi = SVI(self._model, self._guide, optim, loss=loss)

        for i in range(num_epochs):
            elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                input_data, observation_data = input_data.to(
                    self.device), observation_data.to(self.device)
                elbo += svi.step(input_data, observation_data)

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, elbo / num_batch):
                break

        self.model.train(old_training_state)
        return svi

    def _fit(self, data_loader, optim_builder, loss_fn, num_epochs, callback=None, num_particles=1, device=None, lr=False):
        old_training_state = self.model.training
        self.model.train(True)

        with pyro.poutine.trace(param_only=True) as param_capture:
            x, y = next(iter(data_loader))
            x, y = x.to(self.device), y.to(self.device)
            loss = loss_fn.differentiable_loss(self.model, self.guide, x, y)
        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())
        optimizer = optim_builder(params)

        for i in range(num_epochs):
            tot_elbo = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                input_data, observation_data = input_data.to(self.device), observation_data.to(
                    self.device)

                elbo = loss_fn.differentiable_loss(
                    self.model, self.guide, input_data, observation_data)
                elbo.backward()
                # take a step and zero the parameter gradients
                optimizer.step()
                optimizer.zero_grad()

                tot_elbo += elbo

            # the callback can stop training by returning True
            if callback is not None and callback(self, i, tot_elbo / num_batch):
                break

        self.model.train(old_training_state)

        return self

    def guided_forward(self, *args, guide_tr=None, likelihood_guide_tr=None, **kwargs):
        if guide_tr is None:
            guide_tr = poutine.trace(self.guide).get_trace(*args, **kwargs)

        pred = poutine.replay(self.model, trace=guide_tr)(*args, **kwargs)

        return pred

    def predict(self, *input_data, num_predictions=1, aggregate=True, guide_traces=None, likelihood_guide_traces=None):
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        if likelihood_guide_traces is None:
            likelihood_guide_traces = [None] * num_predictions

        preds = []
        scales = []
        with torch.autograd.no_grad():
            for trace, likelihood_trace in zip(guide_traces, likelihood_guide_traces):
                pred = self.guided_forward(
                    *input_data, guide_tr=trace, likelihood_guide_tr=likelihood_trace)
                preds.append(pred)
        predictions = torch.stack(preds)
        return self.likelihood.aggregate_predictions(predictions) if aggregate else predictions
