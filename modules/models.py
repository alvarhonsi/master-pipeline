import pyro
import torch
import torch.nn as nn
from pyro.nn import PyroSample, PyroModule
from pyro.infer import MCMC, SVI, Predictive, Trace_ELBO
import pyro.distributions as dist
import os
import time
from datetime import timedelta
from abc import ABC, abstractmethod

## Skal noise for bias og sigma være 1 eller 10.?
## 10 ser ut til å gjøre at modellene ikke trener riktig

class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        
        prior_loc_w = torch.zeros(out_features, in_features, device=self.device)
        prior_scale_w = torch.ones(out_features, in_features, device=self.device)
        prior_loc_b = torch.zeros(out_features, device=self.device)

        self.linear.weight = PyroSample(dist.Normal(prior_loc_w, prior_scale_w).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(prior_loc_b, 10.).to_event(1))

    def forward(self, x):
        return self.linear(x)


class BayesianRegressor(PyroModule):
    def __init__(self, in_features, out_features, hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device

        def make_layer(in_features, out_features):
            return PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features, device=self.device),
                nn.ReLU()
            )

        if len(hidden_features) == 0:
            self.fc = PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features, device=self.device)
            )
        else:
            self.fc = PyroModule[nn.Sequential](
                make_layer(in_features, hidden_features[0]),
                *[make_layer(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)],
                BayesianLinear(hidden_features[-1], out_features, device=self.device)
            )

    def forward(self, x, y=None):
        out = self.fc(x)
        mu = out.squeeze().to(self.device)

        sigma = pyro.sample("sigma", dist.Uniform(0., 1.)).to(self.device)
        with pyro.plate("data", out.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu



class Regressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device

        def make_layer(in_features, out_features):
            return nn.Sequential(
                nn.Linear(in_features, out_features, device=self.device),
                nn.ReLU()
            )

        if len(hidden_features) == 0:
            self.fc = nn.Sequential(
                nn.Linear(in_features, out_features, device=self.device)
            )
        else:
            self.fc = nn.Sequential(
                make_layer(in_features, hidden_features[0]),
                *[make_layer(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)],
                nn.Linear(hidden_features[-1], out_features, device=self.device)
            )

    def forward(self, x, y=None):
        out = self.fc(x)
        mu = out.squeeze().to(self.device)

        return mu


model_types = {
    "BR": BayesianRegressor,
    "R": Regressor
}