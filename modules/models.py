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

class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        self.linear.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, in_features]).to_event(2))
        self.linear.bias = PyroSample(dist.Normal(0., 10.).expand([out_features]).to_event(1))

    def forward(self, x):
        return self.linear(x)


class BayesianRegressor(PyroModule):
    def __init__(self, in_features, out_features, hidden_features=[]):
        super().__init__()

        def make_layer(in_features, out_features):
            return PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features),
                nn.ReLU()
            )

        if len(hidden_features) == 0:
            self.fc = PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features)
            )
        else:
            self.fc = PyroModule[nn.Sequential](
                make_layer(in_features, hidden_features[0]),
                *[make_layer(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)],
                BayesianLinear(hidden_features[-1], out_features)
            )

    def forward(self, x, y=None):
        out = self.fc(x)
        mu = out.squeeze()

        sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        with pyro.plate("data", out.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu


model_types = {
    "BR": BayesianRegressor
}