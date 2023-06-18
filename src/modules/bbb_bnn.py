from collections import defaultdict
import itertools
from operator import itemgetter
from collections import OrderedDict
import math

import torch
import torch.nn as nn

import pyro
import pyro.nn as pynn
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO, MCMC
from pyro.nn import PyroModule, PyroSample, PyroParam
import pyro.distributions as dist

from . import util


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
    def __init__(self, in_features, out_features, prior, hidden_features=[], device="cpu", name=""):
        super().__init__(name)
        self.device = device
        self.prior = prior
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
        mu = out.squeeze()

        return mu


class GaussianLikelihood(PyroModule):
    def __init__(self, scale, dataset_size, device="cpu", name=""):
        super().__init__(name)
        self.device = device
        print("hei")

        if isinstance(scale, (dist.Distribution)):
            scale = PyroSample(prior=scale)
        else:
            pass
        self._scale = scale
        self.dataset_size = dataset_size

    def forward(self, preds, y=None):

        pred_dist = dist.Normal(preds, self._scale)

        if pred_dist.batch_shape:
            dataset_size = self.dataset_size if self.dataset_size is not None else len(
                preds)
            with pyro.plate(self.data_name+"_plate", subsample=preds, size=dataset_size):
                return pyro.sample(self.data_name, pred_dist, obs=y)
        else:
            dataset_size = self.dataset_size if self.dataset_size is not None else 1
            with pyro.poutine.scale(scale=dataset_size):
                return pyro.sample(self.data_name, pred_dist, obs=y)


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


class BBBGuide(PyroModule):
    def __init__(self, in_features, out_features, prior, hidden_features=[], device="cpu", name=""):
        super().__init__(name)
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        if hidden_features == []:
            self.fc0.weight.mu = PyroParam(torch.full(
                (in_features, out_features), prior.loc, device=self.device))
            self.fc0.weight.rho = PyroParam(torch.full(
                (in_features, out_features), prior.scale, device=self.device))
            self.fc0.bias.mu = PyroParam(torch.full(
                (out_features,), prior.loc, device=self.device))
            self.fc0.bias.roh = PyroParam(torch.full(
                (out_features,), prior.scale, device=self.device))
        else:
            self.fc0.weight.mu = PyroParam(torch.full(
                (in_features, hidden_features[0]), prior.loc, device=self.device))
            self.fc0.weight.rho = PyroParam(torch.full(
                (in_features, hidden_features[0]), prior.scale, device=self.device))
            self.fc0.bias.mu = PyroParam(torch.full(
                (hidden_features[0],), prior.loc, device=self.device))
            self.fc0.bias.rho = PyroParam(torch.full(
                (hidden_features[0],), prior.scale, device=self.device))

    def forward(self, x, y=None):
        if self.hidden_features == []:
            pyro.sample("fc0.linear.weight", ReparameterizedGaussian(
                self.fc0.weight.mu, self.fc0.weight.rho).to_event(2))
            pyro.sample("fc0.linear.bias", ReparameterizedGaussian(
                self.fc0.bias.mu, self.fc0.bias.rho).to_event(1))
        else:
            pyro.sample("fc0.linear.weight", ReparameterizedGaussian(
                self.fc0.weight.mu, self.fc0.weight.rho).to_event(2))
            pyro.sample("fc0.linear.bias", ReparameterizedGaussian(
                self.fc0.bias.mu, self.fc0.bias.rho).to_event(1))
            for i in range(len(self.hidden_features)-1):
                pyro.sample("fc"+str(i+1)+".linear.weight", ReparameterizedGaussian(
                    self.fc[i].weight.mu, self.fc[i].weight.rho).to_event(2))
                pyro.sample("fc"+str(i+1)+".linear.bias", ReparameterizedGaussian(
                    self.fc[i].bias.mu, self.fc[i].bias.rho).to_event(1))
            pyro.sample("fc"+str(len(self.hidden_features))+".linear.weight",
                        ReparameterizedGaussian(self.fc[-1].weight.mu, self.fc[-1].weight.rho).to_event(2))
            pyro.sample("fc"+str(len(self.hidden_features))+".linear.bias",
                        ReparameterizedGaussian(self.fc[-1].bias.mu, self.fc[-1].bias.rho).to_event(1))
