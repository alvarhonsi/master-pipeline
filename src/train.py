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
from pipeline_util import draw_data_samples, save_bnn, load_bnn
from functools import partial
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
import tyxe
import pickle
import dill
import sys
import os
import random


def make_inference_model(config, dataset_config, device=None):
    DEVICE = device if device != None else config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")

    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    PRIOR_LOC = config.getfloat("PRIOR_LOC")
    PRIOR_SCALE = config.getfloat("PRIOR_SCALE")
    OBS_MODEL = config["OBS_MODEL"]
    LIKELIHOOD_SCALE_LOC = config.getfloat("LIKELIHOOD_SCALE_LOC")
    LIKELIHOOD_SCALE = config.getfloat("LIKELIHOOD_SCALE")
    GUIDE_SCALE = config.getfloat("GUIDE_SCALE")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]

    TRAIN_SIZE = dataset_config.getint("TRAIN_SIZE")

    net = get_net(X_DIM, Y_DIM, HIDDEN_FEATURES,
                  activation=nn.ReLU()).to(DEVICE)
    print(net)
    prior_dist = dist.Normal(torch.tensor(
        PRIOR_LOC, device=DEVICE), torch.tensor(PRIOR_SCALE, device=DEVICE))
    prior = tyxe.priors.IIDPrior(prior_dist)

    if OBS_MODEL == "homoskedastic":
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size=TRAIN_SIZE, scale=torch.tensor(LIKELIHOOD_SCALE, device=DEVICE))
        likelihood_guide_builder = None
    elif OBS_MODEL == "homoskedastic_param":
        scale = PyroParam(LIKELIHOOD_SCALE,
                          constraint=dist.constraints.positive)
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size=TRAIN_SIZE, scale=scale)
        likelihood_guide_builder = None
    elif OBS_MODEL == "homoskedastic_gamma":
        scale = dist.Gamma(torch.tensor(LIKELIHOOD_SCALE_LOC, device=DEVICE),
                           torch.tensor(LIKELIHOOD_SCALE, device=DEVICE))
        obs_model = tyxe.likelihoods.HomoskedasticGaussian(
            dataset_size=TRAIN_SIZE, scale=scale)
        likelihood_guide_builder = partial(
            tyxe.guides.AutoNormal, init_scale=GUIDE_SCALE)
    else:
        raise ValueError(
            f"Observation model {OBS_MODEL} not supported. Supported models: homoskedastic, homoskedastic_param, homoskedastic_gamma")
    # Create inference model
    if INFERENCE_TYPE == "svi":
        def init_fn(*args, **kwargs):
            return ag_init.init_to_median(*args, **kwargs).to(DEVICE)
        guide_builder = partial(tyxe.guides.AutoNormal,
                                init_loc_fn=init_fn, init_scale=GUIDE_SCALE)
        bnn = tyxe.VariationalBNN(
            net, prior, obs_model, guide_builder, likelihood_guide_builder=likelihood_guide_builder)
        return bnn
    elif INFERENCE_TYPE == "mcmc":
        kernel_builder = pyro.infer.mcmc.NUTS
        bnn = tyxe.bnn.MCMC_BNN(net, prior, obs_model, kernel_builder)
        return bnn
    else:
        raise ValueError(
            f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")


def train(config, dataset_config, DIR, device=None, print_train=False, reruns=1):

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
    TRAIN_CONTEXT = config["TRAIN_CONTEXT"]
    SVI_PARTICLES = config.getint("SVI_PARTICLES")
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    SAVE_MODEL = config.getboolean("SAVE_MODEL")
    EPOCHS = config.getint("EPOCHS")
    LR = config.getfloat("LR")
    LRD_GAMMA = config.getfloat("LRD_GAMMA")
    LRD_STEPS = config.getint("LRD_STEPS")
    TRAIN_BATCH_SIZE = config.getint("TRAIN_BATCH_SIZE")
    EVAL_BATCH_SIZE = config.getint("EVAL_BATCH_SIZE")
    NUM_DIST_SAMPLES = config.getint("NUM_DIST_SAMPLES")

    VAL_SIZE = dataset_config.getint("VAL_SIZE")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load data
    (x_train, y_train), (x_val, y_val), _, _ = load_data(
        f"{DIR}/datasets/{DATASET}")

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
    subset = np.random.choice(len(train_dataset), size=VAL_SIZE)
    train_subset = torch.utils.data.Subset(
        train_dataset, subset)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True)
    train_subset_dataloader = DataLoader(
        train_subset, batch_size=EVAL_BATCH_SIZE, num_workers=4)
    val_dataloader = DataLoader(
        val_dataset, batch_size=EVAL_BATCH_SIZE, num_workers=4)

    x_t, y_t = next(iter(train_dataloader))
    print(x_t.shape, y_t.shape)

    for run in range(1, reruns + 1):
        # Create model
        bnn = make_inference_model(config, dataset_config, device=DEVICE)

        # RUN TRAINING
        print('Using device: {}'.format(DEVICE))
        print(f'===== Training profile {NAME} - {run} =====')

        pyro.clear_param_store()

        start = time.time()

        if INFERENCE_TYPE == "svi":
            ### SVI ###
            train_stats = {
                "elbos": [],
                "time": 0,
                "val_rmse": [],
                "val_ll": [],
                "train_rmse": [],
                "train_ll": [],
            }

            def callback(bnn, i, e):
                time_elapsed = time.time() - start
                train_stats["elbos"].append(e)

                if i % 10 == 0:
                    val_mse, val_loglikelihood = 0, 0
                    batch_num = 0
                    for num_batch, (input_data, observation_data) in enumerate(iter(val_dataloader), 1):
                        input_data, observation_data = input_data.to(
                            DEVICE), observation_data.to(DEVICE)
                        val_err, val_ll = bnn.evaluate(
                            input_data, observation_data, num_predictions=100, reduction="mean")
                        val_mse += val_err
                        val_loglikelihood += val_ll
                        batch_num = num_batch
                    val_rmse = (val_mse / batch_num).sqrt()
                    val_loglikelihood = val_loglikelihood / batch_num

                    train_mse, train_loglikelihood = 0, 0
                    batch_num = 0
                    for num_batch, (input_data, observation_data) in enumerate(iter(train_subset_dataloader), 1):
                        input_data, observation_data = input_data.to(
                            DEVICE), observation_data.to(DEVICE)
                        train_err, train_ll = bnn.evaluate(
                            input_data, observation_data, num_predictions=100, reduction="mean")
                        train_mse += train_err
                        train_loglikelihood += train_ll
                        batch_num = num_batch
                    train_rmse = (train_mse / batch_num).sqrt()
                    train_loglikelihood = train_loglikelihood / batch_num

                    train_stats["val_rmse"].append(val_rmse.item())
                    train_stats["val_ll"].append(val_loglikelihood.item())
                    train_stats["train_rmse"].append(train_rmse.item())
                    train_stats["train_ll"].append(train_loglikelihood.item())

                    print("[{}] epoch: {} | elbo: {} | train_rmse: {} | val_rmse: {} | val_ll: {}".format(timedelta(
                        seconds=time_elapsed), i, e, round(train_rmse.item(), 4), round(val_rmse.item(), 4), round(val_ll.item(), 4)))

            # optim
            if LRD_GAMMA == 0:
                optim = pyro.optim.Adam(
                    {"lr": LR, "betas": (0.95, 0.999)})
            else:
                # final learning rate will be gamma * initial_lr
                lrd = LRD_GAMMA ** (1 / LRD_STEPS)
                optim = pyro.optim.ClippedAdam(
                    {"lr": LR, "betas": (0.95, 0.999), "lrd": lrd})
            with tyxe.poutine.local_reparameterization():
                bnn.fit(train_dataloader, optim, num_epochs=EPOCHS,
                        callback=callback, device=DEVICE, num_particles=SVI_PARTICLES)
        elif INFERENCE_TYPE == "mcmc":
            ### MCMC ###
            train_stats = {
                "time": 0,
            }

            bnn.fit(train_dataloader, num_samples=MCMC_NUM_SAMPLES,
                    warmup_steps=MCMC_NUM_WARMUP, device=DEVICE)

            # Move mcmc to cpu before diagnostics to avoid memory issues

            with open(f"{DIR}/results/{NAME}/mcmc_diagnostics_{run}.pkl", "wb") as f:
                pickle.dump(bnn._mcmc.diagnostics(), f)

        # Sample likelihood scale
        dummy_input = (torch.zeros(1, X_DIM).to(DEVICE),
                       torch.zeros(1, Y_DIM).to(DEVICE))
        lik_scale = bnn.get_likelihood_scale(
            dummy_input, num_predictions=50)
        train_stats["likelihood"] = {
            "mean": lik_scale[0].item(), "std": lik_scale[1].item()}

        train_stats["time"] = time.time() - start
        print(
            f"Training finished in {timedelta(seconds=train_stats['time'])} seconds")

        if SAVE_MODEL:
            save_bnn(bnn, INFERENCE_TYPE,
                     f"{DIR}/models/{NAME}/checkpoint_{run}.pt")

        with open(f"{DIR}/results/{NAME}/train_stats_{run}.json", "w") as f:
            json.dump(train_stats, f, indent=4)
