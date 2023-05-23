from modules.models import get_net
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.datageneration import load_data, data_functions
from modules.config import read_config
from modules.context import set_default_tensor_type
from modules.plots import plot_comparison_grid
from modules.distributions import DataDistribution
from modules.priors import prior_types
from modules.optimizers import optimizer_types
from modules.guides import guide_types
from modules.loss import loss_types
from tyxe_runfiles.eval import draw_data_samples
from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.nn import PyroModule, PyroSample, PyroParam
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, MCMC, NUTS, HMC, Trace_ELBO, Predictive, TraceMeanField_ELBO
import pyro.infer.autoguide.initialization as ag_init
import pyro.distributions as dist
from datetime import timedelta
import time
import json
import argparse
import tyxe
import tyxe.priors as priors
import tyxe.likelihoods as likelihoods

import functools
from typing import Callable, Optional


def make_inference_model(config, dataset_config, device=None):
    DEVICE = device if device != None else config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")

    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    PRIOR_LOC = config.getfloat("PRIOR_LOC")
    PRIOR_SCALE = config.getfloat("PRIOR_SCALE")
    OBS_MODEL = config["OBS_MODEL"]
    LIKELIHOOD_SCALE = config.getfloat("LIKELIHOOD_SCALE")
    GUIDE_SCALE = config.getfloat("GUIDE_SCALE")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    EPOCHS = config.getint("EPOCHS")
    LR = config.getfloat("LR")

    TRAIN_SIZE = dataset_config.getint("TRAIN_SIZE")
    
    net = get_net(X_DIM, Y_DIM, HIDDEN_FEATURES, activation=nn.ReLU()).to(DEVICE)
    print(net)
    prior_dist = dist.Normal(torch.tensor(PRIOR_LOC, device=DEVICE), torch.tensor(PRIOR_SCALE, device=DEVICE))
    prior = tyxe.priors.IIDPrior(prior_dist)

    if OBS_MODEL == "homoskedastic":
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(dataset_size=TRAIN_SIZE, scale=LIKELIHOOD_SCALE)
        likelihood_guide_builder = None
    elif OBS_MODEL == "homoskedastic_param":
        scale = PyroParam(LIKELIHOOD_SCALE, constraint=dist.constraints.positive)
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(dataset_size=TRAIN_SIZE, scale=scale)
        likelihood_guide_builder = None
    elif OBS_MODEL == "homoskedastic_gamma":
        scale = dist.Gamma(torch.tensor(1., device=DEVICE), torch.tensor(1., device=DEVICE))
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(dataset_size=TRAIN_SIZE, scale=scale)
        likelihood_guide_builder = partial(tyxe.guides.AutoNormal, init_scale=GUIDE_SCALE)
    # Create inference model
    if INFERENCE_TYPE == "svi":
        def init_fn (*args, **kwargs):
            return ag_init.init_to_median(*args, **kwargs).to(DEVICE) 
        guide_builder = partial(tyxe.guides.AutoNormal, init_loc_fn=init_fn, init_scale=GUIDE_SCALE)
        #guide_builder = partial(tyxe.guides.AutoNormal, init_scale=0.01)
        print("train size:", TRAIN_SIZE)
        #obs_model = tyxe.likelihoods.HomoskedasticGaussian(TRAIN_SIZE, scale=PyroParam(torch.tensor(5.), constraint=dist.constraints.positive))
        bnn = tyxe.VariationalBNN(net, prior, obs_model, guide_builder, likelihood_guide_builder=likelihood_guide_builder)
        return bnn
    elif INFERENCE_TYPE == "mcmc":
        kernel_builder = partial(pyro.infer.mcmc.NUTS, step_size=1.)
        #kernel_builder = partial(tyxe.kernels.HMC, step_size=1.)
        bnn = tyxe.bnn.MCMC_BNN(net, prior, obs_model, kernel_builder)
        return bnn
    else:
        raise ValueError(f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")
    


def train(config, dataset_config, DIR, device=None, print_train=False):

    NAME = config["NAME"]
    DEVICE = device if device != None else config["DEVICE"]
    SEED = config.getint("SEED")
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DATASET = config["DATASET"]

    MU = dataset_config.getfloat("MU")
    SIGMA = dataset_config.getfloat("SIGMA")
    DATA_FUNC = dataset_config["DATA_FUNC"]

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    SAVE_MODEL = config.getboolean("SAVE_MODEL")
    EPOCHS = config.getint("EPOCHS")
    LR = config.getfloat("LR")
    TRAIN_BATCH_SIZE = config.getint("TRAIN_BATCH_SIZE")
    NUM_DIST_SAMPLES = config.getint("NUM_DIST_SAMPLES")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    (x_train, y_train), (x_val, y_val), _, _, _ = load_data(f"{DIR}/datasets/{DATASET}")

    # Ready model directory
    if not os.path.exists(f"{DIR}/models/{NAME}"):
        os.mkdir(f"{DIR}/models/{NAME}")

    # Ready results directory
    if not os.path.exists(f"{DIR}/results/{NAME}"):
        os.mkdir(f"{DIR}/results/{NAME}")

    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE)

    x_t, y_t = next(iter(train_dataloader))
    print(x_t.shape, y_t.shape)

    # Create model
    bnn = make_inference_model(config, dataset_config, device=DEVICE)
    
    # RUN TRAINING
    print('Using device: {}'.format(DEVICE))
    print(f'===== Training profile {NAME} =====')

    pyro.clear_param_store()

    start = time.time()

    train_stats = {
        "elbos": [],
        "time": 0,
        "rmse_epoch": [],
        "ll_epoch": [],
    }
    def callback(bnn, i, e):
        time_elapsed = time.time() - start

        mse, loglikelihood = 0, 0
        batch_num = 0
        for num_batch, (input_data, observation_data) in enumerate(iter(val_dataloader), 1):
            err, ll = bnn.evaluate(input_data, observation_data, num_predictions=20, reduction="mean")
            mse += err
            loglikelihood += ll
            batch_num = num_batch
        rmse = (mse / batch_num).sqrt()
        loglikelihood = loglikelihood / batch_num
        
        train_stats["rmse_epoch"].append(rmse.sqrt().item())
        train_stats["ll_epoch"].append(loglikelihood.item())


        if i % 100 == 0:
            print("[{}] epoch: {} | elbo: {} | val_rmse: {} | val_ll: {}".format(timedelta(seconds=time_elapsed), i, e, mse.sqrt().item(), ll.item()))
            
        train_stats["elbos"].append(e)

    if INFERENCE_TYPE == "svi":
        optim = pyro.optim.Adam({"lr": LR})
        with tyxe.poutine.flipout():
            bnn.fit(train_dataloader, optim, num_epochs=EPOCHS, callback=callback, device=DEVICE)
    elif INFERENCE_TYPE == "mcmc":
        bnn.fit(train_dataloader, num_samples=MCMC_NUM_SAMPLES, warmup_steps=MCMC_NUM_WARMUP, device=DEVICE)

    train_stats["time"] = time.time() - start
    print(f"Training finished in {timedelta(seconds=train_stats['time'])} seconds")

    return bnn, train_stats