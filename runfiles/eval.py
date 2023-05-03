from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import difference_mean, difference_std, KL_divergance_normal
from modules.distributions import PredictivePosterior, DataDistribution, NormalPosterior
from modules.context import set_default_tensor_type
from modules.priors import prior_types
from modules.guides import guide_types
from modules.loss import loss_types
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.infer import Predictive, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Normal
from contextlib import contextmanager
import os
import argparse
import json
import time
import datetime

def draw_data_samples(dataloader, num_samples=10):
    xs, ys = [], []
    for x, y in dataloader:
        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    idx = np.random.choice(range(len(x)), num_samples)
    sample_x = x[idx]
    sample_y = y[idx]
    return sample_x, sample_y

def evaluate_error(inference_model, dataloader, num_predictions=100):
    results = {}
    # Evaluate error
    rmse, mae = inference_model.evaluate(dataloader, num_predictions=num_predictions)
    results["rmse"] = np.float64(rmse)
    results["mae"] = np.float64(mae)

    return results


def evaluate_posterior(posterior_samples, data_samples):
    results = {}

    post_std_is_zero = any(np.std(posterior_samples, axis=0) == 0)
    data_std_is_zero = any(np.std(data_samples, axis=0) == 0)

    # Mean kl divergence
    if not post_std_is_zero and not data_std_is_zero:
        kl_div = np.mean(KL_divergance_normal(posterior_samples, data_samples))
        results["kl_div"] = np.float64(kl_div)
    else:
        results["kl_div"] = np.float64(-1)
    # Difference in mean
    mean_diff = np.mean(difference_mean(posterior_samples, data_samples))
    results["mean_diff"] = np.float64(mean_diff)

    # Difference in standard deviation
    std_diff = np.mean(difference_std(posterior_samples, data_samples))
    results["std_diff"] = np.float64(std_diff)

    return results

def load_model(dir, config):
    NAME = config["NAME"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DEVICE = config["DEVICE"]
    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    PRIOR_TYPE = config["PRIOR_TYPE"]
    WEIGHT_LOC = config.getfloat("WEIGHT_LOC")
    WEIGHT_SCALE = config.getfloat("WEIGHT_SCALE")
    BIAS_LOC = config.getfloat("BIAS_LOC")
    BIAS_SCALE = config.getfloat("BIAS_SCALE")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
    SVI_LOSS = config["SVI_LOSS"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    try:
        PRIOR = prior_types[PRIOR_TYPE]
    except KeyError:
        raise ValueError(f"Prior type {PRIOR_TYPE} not supported. Supported types: {prior_types.keys()}")
    try:
        BNN = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    try:
        GUIDE = guide_types[SVI_GUIDE]
    except KeyError:
        raise ValueError(f"Guide type {SVI_GUIDE} not supported. Supported types: {guide_types.keys()}")
    try:
        LOSS = loss_types[SVI_LOSS]
    except KeyError:
        raise ValueError(f"Loss type {SVI_LOSS} not supported. Supported types: {loss_types.keys()}")

    prior = PRIOR(WEIGHT_LOC, WEIGHT_SCALE, BIAS_LOC, BIAS_SCALE)
    model = BNN(X_DIM, Y_DIM, prior, hidden_features = HIDDEN_FEATURES, device=DEVICE)

    if INFERENCE_TYPE == "svi":
        guide = GUIDE(model, device=DEVICE)
        loss = LOSS()
        optim = pyro.optim.Adam({"lr": 1e-3})
        inference_model = SVIInferenceModel(model, prior, guide, optim, loss=loss, device=DEVICE)
    elif INFERENCE_TYPE == "mcmc":
        mcmc_kernel = NUTS(model)
        inference_model = MCMCInferenceModel(model, prior, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS, device=DEVICE)
    else:
        raise ValueError(f"Invalid inference type: {INFERENCE_TYPE}")

    inference_model.load(f"{dir}/models/{NAME}")

    return inference_model

def eval(config, dataset_config, DIR, inference_model=None, device=None):
    NAME = config["NAME"]
    SEED = config.getint("SEED")
    DEVICE = device if device != None else config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DATASET = config["DATASET"]

    DATA_FUNC = dataset_config["DATA_FUNC"]
    MU = dataset_config.getfloat("MU")
    SIGMA = dataset_config.getfloat("SIGMA")

    NUM_X_SAMPLES = config.getint("NUM_X_SAMPLES")
    NUM_DIST_SAMPLES = config.getint("NUM_DIST_SAMPLES")
    EVAL_BATCH_SIZE = config.getint("EVAL_BATCH_SIZE")

    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pyro.clear_param_store()

    # Load data
    (x_train, y_train), _, (x_test, y_test), (x_test_in_domain, y_test_in_domain), (x_test_out_domain, y_test_out_domain) = load_data(f"{DIR}/datasets/{DATASET}", load_val=False)

    # Make dataset
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    test_in_domain_dataset = TensorDataset(x_test_in_domain, y_test_in_domain)
    test_out_domain_dataset = TensorDataset(x_test_out_domain, y_test_out_domain)

    # Make dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_in_domain_dataloader = DataLoader(test_in_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_out_domain_dataloader = DataLoader(test_out_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    
    # Load model
    inference_model = load_model(DIR, config) if inference_model is None else inference_model

    # Ready results directory
    if not os.path.exists(f"{DIR}/results/{NAME}"):
        os.mkdir(f"{DIR}/results/{NAME}")

    start = time.time()
    print(f"using device: {DEVICE}")
    print(f"====== evaluating profile {NAME} ======")

    # Draw samples from relevant distributions
    
    # Ready samples directory
    if not os.path.exists(f"{DIR}/results/{NAME}/samples"):
        os.mkdir(f"{DIR}/results/{NAME}/samples")

    # samp_x: (NUM_X_SAMPLES, X_DIM), samp_y: (NUM_X_SAMPLES)
    test_x_sample, _ = draw_data_samples(test_dataloader, NUM_X_SAMPLES)
    test_in_domain_x_sample, _ = draw_data_samples(test_in_domain_dataloader, NUM_X_SAMPLES)
    test_out_domain_x_sample, _ = draw_data_samples(test_out_domain_dataloader, NUM_X_SAMPLES)

    np.savetxt(f"{DIR}/results/{NAME}/samples/test_x_sample.csv", test_x_sample, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/test_in_domain_x_sample.csv", test_in_domain_x_sample, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/test_out_domain_x_sample.csv", test_out_domain_x_sample, delimiter=",")
    
    #Sample true distribution from data
    # data_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    func = data_functions[DATA_FUNC]
    data_dist = DataDistribution(func, MU, SIGMA, test_x_sample)
    data_in_domain_dist = DataDistribution(func, MU, SIGMA, test_in_domain_x_sample)
    data_out_domain_dist = DataDistribution(func, MU, SIGMA, test_out_domain_x_sample)

    data_samples = data_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_in_domain_samples = data_in_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_out_domain_samples = data_out_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.savetxt(f"{DIR}/results/{NAME}/samples/data_samples.csv", data_samples, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/data_in_domain_samples.csv", data_in_domain_samples, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/data_out_domain_samples.csv", data_out_domain_samples, delimiter=",")
    

    #Sample posterior distribution from model
    # pred_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    pred_samples = inference_model.predict(test_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
    pred_in_domain_samples = inference_model.predict(test_in_domain_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
    pred_out_domain_samples = inference_model.predict(test_out_domain_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.savetxt(f"{DIR}/results/{NAME}/samples/predictive_samples.csv", pred_samples, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/predictive_in_domain_samples.csv", pred_in_domain_samples, delimiter=",")
    np.savetxt(f"{DIR}/results/{NAME}/samples/predictive_out_domain_samples.csv", pred_out_domain_samples, delimiter=",")

    # Sanity Checks
    train_x_sample, train_y_sample = draw_data_samples(train_dataloader, NUM_X_SAMPLES)
    train_data_dist = DataDistribution(func, MU, SIGMA, train_x_sample)
    train_data_samples = train_data_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    train_pred_samples = inference_model.predict(train_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
    plot_comparison_grid(train_pred_samples, train_data_samples, grid_size=(3,3), figsize=(20,20), kl_div=True, title="Posterior samples - Train", plot_mean=True, save_path=f"{DIR}/results/{NAME}/train_sanity.png")



    plot_comparison_grid(pred_samples, data_samples, grid_size=(3,3), figsize=(20,20), kl_div=True, title="Posterior samples - Test", plot_mean=True, save_path=f"{DIR}/results/{NAME}/test_sanity.png")
    plot_comparison_grid(pred_in_domain_samples, data_in_domain_samples, grid_size=(3,3), figsize=(20,20), kl_div=True, title="Posterior samples - In Domain", plot_mean=True, save_path=f"{DIR}/results/{NAME}/test_in_domain_sanity.png")
    plot_comparison_grid(pred_out_domain_samples, data_out_domain_samples, grid_size=(3,3), figsize=(20,20), kl_div=True, title="Posterior samples - Out of Domain", plot_mean=True, save_path=f"{DIR}/results/{NAME}/test_out_domain_sanity.png")
    

    # Evaluate

    results = {}

    test_error = evaluate_error(inference_model, test_dataloader, num_predictions=100)
    in_domain_error = evaluate_error(inference_model, test_in_domain_dataloader, num_predictions=100)
    out_domain_error = evaluate_error(inference_model, test_out_domain_dataloader, num_predictions=100)
    results["test_error"] = test_error
    results["in_domain_error"] = in_domain_error
    results["out_domain_error"] = out_domain_error

    results["predictive"] = evaluate_posterior(pred_samples, data_samples)
    results["predictive_in_domain"] = evaluate_posterior(pred_in_domain_samples, data_in_domain_samples)
    results["predictive_out_domain"] = evaluate_posterior(pred_out_domain_samples, data_out_domain_samples)

    # Save results
    with open(f"{DIR}/results/{NAME}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Get time and format to HH:MM:SS
    elapsed_time = str(datetime.timedelta(seconds=time.time() - start))
    print(f"Eval done in {elapsed_time}")
