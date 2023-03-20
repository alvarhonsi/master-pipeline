from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.datageneration import load_data
from modules.config import read_config
from modules.context import set_default_tensor_type
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, MCMC, NUTS, HMC, Trace_ELBO, Predictive
from pyro.infer.autoguide.initialization import init_to_feasible, init_to_median
from datetime import timedelta
import time
import json
import argparse

import functools
from typing import Callable, Optional


def train(config, DIR, device=None):

    NAME = config["NAME"]
    DEVICE = device if device != None else config["DEVICE"]
    SEED = config.getint("SEED")
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DATASET = config["DATASET"]

    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
    SVI_ELBO = config["SVI_ELBO"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    SAVE_MODEL = config.getboolean("SAVE_MODEL")
    EPOCHS = config.getint("EPOCHS")
    LR = config.getfloat("LR")
    TRAIN_BATCH_SIZE = config.getint("TRAIN_BATCH_SIZE")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    (x_train, y_train), (x_val, y_val), _, _, _ = load_data(f"{DIR}/datasets/{DATASET}")

    # Ready model directory
    if not os.path.exists(f"{DIR}/models/{NAME}"):
        os.mkdir(f"{DIR}/models/{NAME}")

    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=3)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=3)

    # Create model
    try:
        BNN = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    model = BNN(X_DIM, Y_DIM, hidden_features=HIDDEN_FEATURES, device=DEVICE)

    # Create inference model
    if INFERENCE_TYPE == "svi":
        def init_func(*args):
            return init_to_median(*args).to(DEVICE)

        guide = AutoDiagonalNormal(model, init_loc_fn=init_func)
        optim = pyro.optim.Adam({"lr": LR})
        inference_model = SVIInferenceModel(model, guide, optim, EPOCHS, device=DEVICE)
    elif INFERENCE_TYPE == "mcmc":
        #mcmc_kernel = NUTS(model, adapt_step_size=True, jit_compile=True)
        mcmc_kernel = NUTS(model, adapt_step_size=True)
        #mcmc_kernel = HMC(model, adapt_step_size=True)
        inference_model = MCMCInferenceModel(model, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, 
        num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS, device=DEVICE)
    else:
        raise ValueError(f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")

    # RUN TRAINING
    print('Using device: {}'.format(DEVICE))
    print(f'===== Training profile {NAME} =====')
    train_stats = inference_model.fit(train_dataloader, val_dataloader)

    if SAVE_MODEL:
        inference_model.save(f"{DIR}/models/{NAME}")
        with open(f"{DIR}/models/{NAME}/train_stats.json", "w") as json_file:
            json.dump(train_stats, json_file, indent=4)

    return inference_model