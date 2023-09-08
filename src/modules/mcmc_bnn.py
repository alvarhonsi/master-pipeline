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


class MCMC_BNN(PyroModule):
    def __init__(self, model, likelihood, kernel_builder, device="cpu"):
        super().__init__()
        self.device = device
        self.model = model
        self.likelihood = likelihood
        self.kernel = kernel_builder(self._model)
        self._mcmc = None

    @pynn.pyro_method
    def _model(self, x, obs=None):
        out = self.model(*_as_tuple(x))
        self.likelihood(out, obs=obs)
        return out

    def fit(self, data_loader, num_samples, num_warmup, num_chains, batch_data=False, mp_context=None, device=None):
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
                          warmup_steps=num_warmup, num_chains=num_chains, mp_context=mp_context)
        self._mcmc.run(input_data, observation_data)

        return self._mcmc

    def predict(self, *input_data, num_predictions=1, aggregate=True):
        if self._mcmc is None:
            raise RuntimeError(
                "Call .fit to run MCMC and obtain samples from the posterior first.")

        preds = []
        scales = []
        weight_samples = self._mcmc.get_samples(num_samples=num_predictions)
        print(weight_samples["likelihood._scale"])
        with torch.no_grad():
            for i in range(num_predictions):
                weights = {name: sample[i]
                           for name, sample in weight_samples.items()}
                preds.append(poutine.condition(
                    self._model, weights)(*input_data))
                # sample scale from distribution
                scales.append(poutine.condition(
                    self.likelihood.get_scale, weights)())
        predictions = torch.stack(preds)
        scales = torch.stack(scales)
        print(scales)
        return self.likelihood.aggregate_predictions((predictions, scales)) if aggregate else predictions
