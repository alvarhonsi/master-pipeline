from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
from pyro.infer import Predictive
from pyro.distributions import Normal

class SamplableDistribution(ABC):
    """A class that represents a sampalable posterior distribution."""

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the distribution."""
        raise NotImplementedError

class NormalPosterior(SamplableDistribution):
    """A class that represents a normal posterior distribution."""

    def __init__(self, mean, std, x):
        self.mean = mean
        self.std = std
        self.x = x

    def sample(self, num_samples) -> torch.Tensor:
        return Normal(self.mean, self.std).sample(sample_shape=(num_samples, self.x.shape[0]))

class PredictivePosterior(SamplableDistribution):
    """A class that represents a predictive posterior distribution."""

    def __init__(self, model, guide, x):
        self.model = model
        self.guide = guide
        self.x = x

    def sample(self, num_samples) -> torch.Tensor:
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))
        samples = predictive(self.x)

        y_samples = samples["obs"]
        mean_samples = samples["_RETURN"]
        if mean_samples.dim() == 1: # Mean is only 1 dimensional if there is only one x sample
            mean_samples = mean_samples.unsqueeze(1)

        # rotate y_samples
        y_samples = torch.transpose(y_samples, 0, 1)
        mean_samples = torch.transpose(mean_samples, 0, 1)

        return y_samples

class DataDistribution(SamplableDistribution):
    """A class that represents a data distribution."""

    def __init__(self, func, mu, std, x):
        self.func = func
        self.mu = mu
        self.std = std
        self.x = x

    def sample(self, num_samples) -> torch.Tensor:
        ys = self.func(self.x).unsqueeze(1) # calc y values
        distributions = Normal(self.mu, self.std) # create normal distribution with mean and std
        noise = distributions.sample(sample_shape=(self.x.shape[0], num_samples)) # sample noise [x_samples, num_dist_samples]
        samples = torch.add(noise, ys) # add y values to noise -> [x_samples, num_dist_samples]
        return samples