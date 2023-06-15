from modules.models import model_types
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
from runfiles.eval import draw_data_samples
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, MCMC, NUTS, HMC, Trace_ELBO, Predictive, TraceMeanField_ELBO
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_median
from datetime import timedelta
import time
import json
import argparse

import functools
from typing import Callable, Optional

def make_inference_model(config, device=None):
    DEVICE = device if device != None else config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")

    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    PRIOR_TYPE = config["PRIOR_TYPE"]
    WEIGHT_LOC = config.getfloat("WEIGHT_LOC")
    WEIGHT_SCALE = config.getfloat("WEIGHT_SCALE")
    BIAS_LOC = config.getfloat("BIAS_LOC")
    BIAS_SCALE = config.getfloat("BIAS_SCALE")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
    SVI_OPTIMIZER = config["SVI_OPTIMIZER"]
    SVI_LOSS = config["SVI_LOSS"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    EPOCHS = config.getint("EPOCHS")
    LR = config.getfloat("LR")


    try:
        PRIOR = prior_types[PRIOR_TYPE]
    except KeyError:
        raise ValueError(f"Prior type {PRIOR_TYPE} not supported. Supported types: {prior_types.keys()}")
    try:
        BNN = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    try:
        OPTIMIZER = optimizer_types[SVI_OPTIMIZER]
    except KeyError:
        raise ValueError(f"Optimizer type {SVI_OPTIMIZER} not supported. Supported types: {optimizer_types.keys()}")
    try:
        GUIDE = guide_types[SVI_GUIDE]
    except KeyError:
        raise ValueError(f"Guide type {SVI_GUIDE} not supported. Supported types: {guide_types.keys()}")
    try:
        LOSS = loss_types[SVI_LOSS]
    except KeyError:
        raise ValueError(f"Loss type {SVI_LOSS} not supported. Supported types: {loss_types.keys()}")
    
    
    prior = PRIOR(WEIGHT_LOC, WEIGHT_SCALE, BIAS_LOC, BIAS_SCALE)
    model = BNN(X_DIM, Y_DIM, prior, hidden_features=HIDDEN_FEATURES, device=DEVICE)

    # Create inference model
    if INFERENCE_TYPE == "svi":
        guide = GUIDE(model, device=DEVICE)
        print(guide)
        optim = OPTIMIZER({"lr": LR})
        loss = LOSS(num_particles=10)
        inference_model = SVIInferenceModel(model, prior, guide, optim, EPOCHS, loss=loss, device=DEVICE)
        return inference_model
    elif INFERENCE_TYPE == "mcmc":
        #mcmc_kernel = NUTS(model, adapt_step_size=True, jit_compile=True)
        mcmc_kernel = NUTS(model, adapt_step_size=True)
        #mcmc_kernel = HMC(model, adapt_step_size=True)
        inference_model = MCMCInferenceModel(model, prior, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, 
        num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS, device=DEVICE)
        return inference_model
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

    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    PRIOR_TYPE = config["PRIOR_TYPE"]
    WEIGHT_LOC = config.getfloat("WEIGHT_LOC")
    WEIGHT_SCALE = config.getfloat("WEIGHT_SCALE")
    BIAS_LOC = config.getfloat("BIAS_LOC")
    BIAS_SCALE = config.getfloat("BIAS_SCALE")
    SIGMA_CONCENTRATION = config.getfloat("SIGMA_CONCENTRATION")
    SIGMA_RATE = config.getfloat("SIGMA_RATE")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
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

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=3)

    # Create model
    inference_model = make_inference_model(config, device=DEVICE)
    
    pyro.clear_param_store()
    
    inference_model.initialize()

    # Sanity check
    if INFERENCE_TYPE == "svi" and False:
        func = data_functions[DATA_FUNC]
        train_x_sample, train_y_sample = draw_data_samples(train_dataloader, 10)
        train_data_dist = DataDistribution(func, MU, SIGMA, train_x_sample)
        train_data_samples = train_data_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()

        inference_model.svi.step(train_x_sample[-1].to(DEVICE), train_y_sample[-1].to(DEVICE)) # Must pass at least one sample to initialize the guide
        train_pred_samples = inference_model.predict(train_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
        plot_comparison_grid(train_pred_samples, train_data_samples, grid_size=(3,3), figsize=(20,20), kl_div=True, title="Posterior samples - Initialized Train", plot_mean=True, save_path=f"{DIR}/results/{NAME}/init_train_sanity.png")

        pyro.clear_param_store()


    # RUN TRAINING
    print('Using device: {}'.format(DEVICE))
    print(f'===== Training profile {NAME} =====')
    train_stats = inference_model.fit(train_dataloader, val_dataloader, print_every=50 if print_train else None)

    print(f"Training finished in {timedelta(seconds=train_stats['time'])} seconds")

    if SAVE_MODEL:
        inference_model.save(f"{DIR}/models/{NAME}")
        with open(f"{DIR}/models/{NAME}/train_stats.json", "w") as json_file:
            json.dump(train_stats, json_file, indent=4)

    return inference_model