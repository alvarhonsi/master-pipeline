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
from modules.priors import NormalPrior

## Skal noise for bias og sigma være 1 eller 10.?
## 10 ser ut til å gjøre at modellene ikke trener riktig

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x)
        return x

class BayesianLinear(PyroModule):
    def __init__(self, in_features, out_features, prior, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        
        prior_loc_w = torch.full_like(self.linear.weight, prior.weight_loc, device=self.device)
        prior_scale_w = torch.full_like(self.linear.weight, prior.weight_scale, device=self.device)
        prior_loc_b = torch.full_like(self.linear.bias, prior.bias_loc, device=self.device)
        prior_scale_b = torch.full_like(self.linear.bias, prior.bias_scale, device=self.device)

        prior_dist = prior.prior_dist
        self.linear.weight = PyroSample(prior_dist(prior_loc_w, prior_scale_w).to_event(2))
        self.linear.bias = PyroSample(prior_dist(prior_loc_b, prior_scale_b).to_event(1))

    def forward(self, x):
        return self.linear(x)
    
class BayesianFixedLinear(PyroModule):
    def __init__(self, in_features, out_features, prior, device="cpu"):
        super().__init__()
        self.device = device
        self.linear = PyroModule[nn.Linear](in_features, out_features)
        
        prior_loc_w = torch.full_like(self.linear.weight, prior.weight_loc, device=self.device)
        prior_scale_w = torch.full_like(self.linear.weight, 0.1, device=self.device)
        prior_loc_b = torch.full_like(self.linear.bias, prior.bias_loc, device=self.device)
        prior_scale_b = torch.full_like(self.linear.bias, 0.1, device=self.device)

        prior_dist = prior.prior_dist
        self.linear.weight = PyroSample(prior_dist(prior_loc_w, prior_scale_w).to_event(2))
        self.linear.bias = PyroSample(prior_dist(prior_loc_b, prior_scale_b).to_event(1))

    def forward(self, x):
        return self.linear(x)


class BayesianRegressorV1(PyroModule):
    def __init__(self, in_features, out_features, prior=NormalPrior(), hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device
        self.prior = prior
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        def make_layer(in_features, out_features):
            return PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features, self.prior, device=self.device),
                nn.ReLU()
                #nn.LeakyReLU(0.1)
            )

        if len(hidden_features) == 0:
            self.fc = PyroModule[nn.Sequential](
                BayesianLinear(in_features, out_features, self.prior, device=self.device)
            )
        else:
            self.fc = PyroModule[nn.Sequential](
                make_layer(in_features, hidden_features[0]),
                *[make_layer(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)],
                BayesianLinear(hidden_features[-1], out_features, self.prior, device=self.device)
            )

    def forward(self, x, y=None):
        out = self.fc(x)
        mu = out.squeeze().to(self.device)

        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0., device=self.device), torch.tensor(10., device=self.device)))
        #sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        #sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))
        with pyro.plate("data", out.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu
    
class BayesianRegressorV2(PyroModule):
    def __init__(self, in_features, out_features, prior=NormalPrior(), hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device
        self.prior = prior
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        if len(hidden_features) == 0:
            self.fc0 = PyroModule[nn.Linear](in_features, out_features)
            self.fc0.weight = PyroSample(dist.Normal(torch.full_like(self.linear.weight, prior.weight_loc, device=self.device), torch.full_like(self.linear.weight, prior.weight_scale, device=self.device)).to_event(2))
            self.fc0.bias = PyroSample(dist.Normal(torch.full_like(self.linear.bias, prior.bias_loc, device=self.device), prior_scale_b = torch.full_like(self.linear.bias, 0.1, device=self.device)).to_event(1))
        else:
            for i in range(len(hidden_features)):
                if i == 0:
                    setattr(self, "fc{}".format(i), PyroModule[nn.Linear](in_features, hidden_features[i]))
                    getattr(self, "fc{}".format(i)).weight = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(i)).weight, prior.weight_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(i)).weight, prior.weight_scale, device=self.device)).to_event(2))
                    getattr(self, "fc{}".format(i)).bias = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(i)).bias, prior.bias_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(i)).bias, prior.bias_scale, device=self.device)).to_event(1))
                else:
                    setattr(self, "fc{}".format(i), PyroModule[nn.Linear](hidden_features[i-1], hidden_features[i]))
                    getattr(self, "fc{}".format(i)).weight = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(i)).weight, prior.weight_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(i)).weight, prior.weight_scale, device=self.device)).to_event(2))
                    getattr(self, "fc{}".format(i)).bias = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(i)).bias, prior.bias_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(i)).bias, prior.bias_scale, device=self.device)).to_event(1))
            setattr(self, "fc{}".format(len(hidden_features)), PyroModule[nn.Linear](hidden_features[-1], out_features))
            getattr(self, "fc{}".format(len(hidden_features))).weight = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(len(hidden_features))).weight, prior.weight_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(len(hidden_features))).weight, prior.weight_scale, device=self.device)).to_event(2))
            getattr(self, "fc{}".format(len(hidden_features))).bias = PyroSample(dist.Normal(torch.full_like(getattr(self, "fc{}".format(len(hidden_features))).bias, prior.bias_loc, device=self.device), torch.full_like(getattr(self, "fc{}".format(len(hidden_features))).bias, prior.bias_scale, device=self.device)).to_event(1))

        self.A = nn.ReLU()

    def forward(self, x, y=None):
        if len(self.hidden_features) == 0:
            out = self.fc0(x)
        else:
            out = x
            for i in range(len(self.hidden_features)):
                out = getattr(self, "fc{}".format(i))(out)
                out = self.A(out)
            out = getattr(self, "fc{}".format(len(self.hidden_features)))(out)
 
        mu = out.squeeze().to(self.device)

        #sigma = pyro.sample("sigma", dist.LogNormal(torch.tensor(0., device=self.device), torch.tensor(1.0, device=self.device)))
        #sigma = pyro.sample("sigma", dist.Uniform(0., 10.))
        #sigma = pyro.sample("sigma", dist.LogNormal(0., 1.))

        sigma = pyro.sample("sigma", dist.Uniform(torch.tensor(0., device=self.device), torch.tensor(1.0, device=self.device)))
        with pyro.plate("data", out.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mu, sigma), obs=y)
        return mu

    
class BayesianRegressorFixed(PyroModule):
    def __init__(self, in_features, out_features, prior=NormalPrior(), hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device
        self.prior = prior
        self.in_features = in_features
        self.out_features = out_features

        def make_layer(in_features, out_features):
            return PyroModule[nn.Sequential](
                BayesianFixedLinear(in_features, out_features, self.prior, device=self.device),
                nn.ReLU()
            )

        if len(hidden_features) == 0:
            self.fc = PyroModule[nn.Sequential](
                BayesianFixedLinear(in_features, out_features, self.prior, device=self.device)
            )
        else:
            self.fc = PyroModule[nn.Sequential](
                make_layer(in_features, hidden_features[0]),
                *[make_layer(hidden_features[i], hidden_features[i+1]) for i in range(len(hidden_features)-1)],
                BayesianFixedLinear(hidden_features[-1], out_features, self.prior, device=self.device)
            )


    def forward(self, x, y=None):
        x = x.to(self.device)
        out = self.fc(x)
        mu = out.squeeze().to(self.device)

        with pyro.plate("data", out.shape[0], device=self.device):
            obs = pyro.sample("obs", dist.Normal(mu, torch.full_like(mu, 0.1, device=self.device)), obs=y)

        return mu



class Regressor(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features

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
    

model_types = {
    "BR": BayesianRegressorV1,
    "BR2": BayesianRegressorV2,
    "R": Regressor,
    "BRF": BayesianRegressorFixed
}