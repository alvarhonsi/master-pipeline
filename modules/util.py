from abc import ABC, abstractmethod
import torch
from pyro.infer import Predictive

class SamplableDistribution(ABC):
    """A class that represents a sampalable posterior distribution."""

    @abstractmethod
    def sample(self, x, num_samples):
        """Sample from the distribution."""
        raise NotImplementedError

class NormalPosterior(SamplableDistribution):
    """A class that represents a normal posterior distribution."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def sample(self, x, num_samples):
        return torch.distributions.Normal(self.mean, self.std).sample((num_samples,))

class PredictivePosterior(SamplableDistribution):
    """A class that represents a predictive posterior distribution."""

    def __init__(self, model, guide, x):
        self.model = model
        self.guide = guide
        self.x = x

    def sample(self, num_samples):
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=("obs", "_RETURN"))
        samples = predictive(self.x)


        return self.model.sample(self.guide, self.predictive_samples, num_samples)