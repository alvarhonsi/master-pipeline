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
import pyro.distributions as dist
from datetime import timedelta
import time
import json
import argparse
import tyxe

import functools
from typing import Callable, Optional


def train(config, DIR):

    NAME = config["NAME"]
    DEVICE = config["DEVICE"]
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

    # Ready data directory
    if not os.path.exists(f"{DIR}/{NAME}"):
        os.mkdir(f"{DIR}/{NAME}")

    # Load data
    (x_train, y_train), (x_val, y_val), _, _, _ = load_data(f"{DIR}/datasets/{DATASET}")

    # Ready model directory
    if not os.path.exists(f"{DIR}/{NAME}/model"):
        os.mkdir(f"{DIR}/{NAME}/model")

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
        MODEL = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    model = MODEL(X_DIM, Y_DIM, hidden_features=HIDDEN_FEATURES, device=DEVICE)

    # Create inference model
    if INFERENCE_TYPE == "svi":
        def init_func(*args):
            return init_to_median(*args).to(DEVICE)

        start = time.time()
        def callback(b, i, avg_elbo):
            avg_err, avg_ll = 0., 0.
            b.eval()
            for x, y in iter(val_dataloader):
                err, ll = b.evaluate(x.to(DEVICE), y.to(DEVICE), num_predictions=100, reduction="mean")
                avg_err += err / len(val_dataloader.sampler)
                avg_ll += ll / len(val_dataloader.sampler)
            
            td = timedelta(seconds=round(time.time() - start))
            if i % 10 == 0:
                print(f"[{td}] Epoch: {i} | ELBO: {avg_elbo} | val error: {avg_err} | ll: {avg_ll}")
            b.train()
        
        optim = pyro.optim.Adam({"lr": 1e-3})

        prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE)))
        likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=len(train_dataloader.sampler))
        guide = tyxe.guides.AutoNormal
        bnn = tyxe.VariationalBNN(model, prior, likelihood, guide)
    elif INFERENCE_TYPE == "mcmc":
        prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE)))
        likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=len(train_dataloader.sampler))
        kernel = pyro.infer.mcmc.NUTS
        bnn = tyxe.MCMC_BNN(model, prior, likelihood, kernel)
    else:
        raise ValueError(f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")

    # RUN TRAINING
    print('Using device: {}'.format(DEVICE))
    print(f'===== Training profile {NAME} =====')
    if INFERENCE_TYPE == "svi":
        bnn.fit(train_dataloader, optim, num_epochs=EPOCHS, callback=callback)
    elif INFERENCE_TYPE == "mcmc":
        bnn.fit(train_dataloader, MCMC_NUM_SAMPLES)

    #if SAVE_MODEL:
        #inference_model.save(f"{DIR}/{NAME}/model")
        #with open(f"{DIR}/{NAME}/model/train_stats.json", "w") as json_file:
            #json.dump(train_stats, json_file, indent=4)

    return bnn