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
import pyro.distributions as dist
import pyro.infer.autoguide.initialization as ag_init

from functools import partial

from . import util


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

    def sample_activation(self, x):
        print(x.type())
        print(self.loc_w.type())
        _w_mu = torch.matmul(x, self.loc_w)
        _w_var = torch.matmul(x.pow(2), self.scale_w.pow(2))
        _w_std = torch.sqrt(1e-06 + F.softplus(_w_var))
        _w_eps = torch.randn_like(_w_std)
        _w_out = _w_mu + _w_std * _w_eps
        _b_out = self.loc_b + self.scale_b * torch.randn_like(self.loc_b)

        return _w_out + _b_out

    def forward(self, x):
        out = pyro.sample("activation", partial(self.sample_activation, x))
        return out


class BayesianLayer(pynn.PyroModule):
    def __init__(self, in_features, out_features, prior, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = PyroModule[nn.Linear](in_features, out_features)

        prior_loc_w = torch.full_like(
            self.linear.weight, prior.loc, device=self.device)
        prior_scale_w = torch.full_like(
            self.linear.weight, prior.scale, device=self.device)
        prior_loc_b = torch.full_like(
            self.linear.bias, prior.loc, device=self.device)
        prior_scale_b = torch.full_like(
            self.linear.bias, prior.scale, device=self.device)

        self.linear.weight = PyroSample(dist.Normal(
            prior_loc_w, prior_scale_w).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(
            prior_loc_b, prior_scale_b).to_event(1))

    def forward(self, x):
        return self.linear(x)


class BayesianNN(PyroModule):
    def __init__(self, in_features, out_features, prior, likelihood, hidden_features=[], device="cpu", name=""):
        super().__init__(name)
        self.device = device
        self.prior = prior
        self.likelihood = likelihood
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        mods = OrderedDict()

        if hidden_features == []:
            mods["fc0"] = BayesianLayer(
                in_features, out_features, self.prior, device=self.device)
        else:
            mods["fc0"] = BayesianLayer(
                in_features, hidden_features[0], self.prior, device=self.device)
            mods["act0"] = nn.ReLU()
            for i in range(len(hidden_features)-1):
                mods["fc"+str(i+1)] = BayesianLayer(hidden_features[i],
                                                    hidden_features[i+1], self.prior, device=self.device)
                mods["act"+str(i+1)] = nn.ReLU()
            mods["fc"+str(len(hidden_features))] = BayesianLayer(
                hidden_features[-1], out_features, self.prior, device=self.device)

        self.fc = PyroModule[nn.Sequential](mods)

    def forward(self, x, y=None):
        out = self.fc(x)
        mu = out

        return mu


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


class ReparameterizedGaussian:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        # log1p <- ln(1 + input)
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.mu.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()


class NormalGuide(PyroModule):
    def __init__(self, model, device="cpu", name="", init_fn=ag_init.init_to_median):
        super().__init__(name)
        self.device = device
        self.model = model

        if isinstance(self.model, BayesianNN):
            dummy_x = torch.zeros(
                1, self.model.in_features, device=self.device)
        elif isinstance(self.model, GaussianLikelihood):
            dummy_x = torch.zeros(1, 1, device=self.device)

        self.sample_sites = pyro.poutine.trace(
            self.model).get_trace(dummy_x).nodes.values()

        for site in self.sample_sites:
            if "weight" in site["name"]:
                std = site["fn"].stddev.detach().shape
                loc = init_fn(site)
                # set self attr of pyro param
                attname = self.getattname(site["name"])
                setattr(self, attname+"_loc", PyroParam(
                    loc))
                setattr(self, attname+"_rho", PyroParam(
                    torch.zeros(std), constraint=dist.constraints.positive))
                print(getattr(self, attname+"_loc").shape)

    def getattname(self, name):
        return name.replace(".", "_")
