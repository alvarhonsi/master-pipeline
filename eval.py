from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import KLdivergence, difference_mean, difference_std, difference_var
from modules.distributions import PredictivePosterior, DataDistribution, NormalPosterior
from modules.context import set_default_tensor_type
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

def eval(config, dataset_config, DIR, inference_model=None):
    NAME = config["NAME"]
    SEED = config.getint("SEED")
    DEVICE = config["DEVICE"]
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")
    DATASET = config["DATASET"]

    DATA_FUNC = dataset_config["DATA_FUNC"]
    MU = dataset_config.getint("MU")
    SIGMA = dataset_config.getint("SIGMA")

    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
    SVI_ELBO = config["SVI_ELBO"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

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
    _, (x_val, y_val), (x_test, y_test) = load_data(f"{DIR}/datasets/{DATASET}", load_train=False)

    # Make dataset
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    
    # Load model
    if inference_model is None:
        BNN = model_types[MODEL_TYPE]
        model = BNN(X_DIM, Y_DIM, HIDDEN_FEATURES, device=DEVICE)

        if INFERENCE_TYPE == "svi":
            guide = AutoDiagonalNormal(model)
            optim = pyro.optim.Adam({"lr": 1e-3})
            inference_model = SVIInferenceModel(model, guide, optim, device=DEVICE)
        elif INFERENCE_TYPE == "mcmc":
            mcmc_kernel = NUTS(model)
            inference_model = MCMCInferenceModel(model, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS, device=DEVICE)
        else:
            raise ValueError(f"Invalid inference type: {INFERENCE_TYPE}")

        inference_model.load(f"{DIR}/{NAME}/model")

    # Ready results directory
    if not os.path.exists(f"{DIR}/{NAME}/results"):
        os.mkdir(f"{DIR}/{NAME}/results")

    print(f"using device: {DEVICE}")
    print(f"====== evaluating profile {NAME} ======")

    # Draw samples from relevant distributions

    # samp_x: (NUM_X_SAMPLES, X_DIM), samp_y: (NUM_X_SAMPLES)
    samp_x, _ = draw_data_samples(test_dataloader, NUM_X_SAMPLES)
    

    #Baseline normal distribution
    # normal_samples: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    base_dist = NormalPosterior(0, 1, samp_x)
    normal_samples = base_dist.sample(NUM_DIST_SAMPLES)
    

    #Sample true distribution from data
    # data_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    func = data_functions[DATA_FUNC]
    data_dist = DataDistribution(func, MU, SIGMA, samp_x)
    data_samp = data_dist.sample(NUM_DIST_SAMPLES)
    

    #Sample posterior distribution from model
    # pred_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    pred_samp = inference_model.predict(samp_x, NUM_DIST_SAMPLES)
    


    #plot_comparison(pred_samp[:, 0], data_samp[:, 0], f"{RESULT_PATH}/{name}/comparison.png")
    norm_samp = normal_samples.cpu().detach().numpy()
    data_samp = data_samp.cpu().detach().numpy()
    pred_samp = pred_samp.cpu().detach().numpy()
    #plot_comparison_grid(data_samp, pred_samp, grid_size=3, save_path=f"{RESULTS_PATH}/{NAME}/comparison_grid.png")

    # Save samples
    np.savetxt(f"{DIR}/{NAME}/results/baseline_normal_samples.csv", norm_samp, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/data_samples.csv", data_samp, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/predictive_samples.csv", pred_samp, delimiter=",")

    # Evaluate

    results = {}

    test_rmse = inference_model.evaluate(test_dataloader)
    results["test_rmse"] = np.float64(test_rmse)

    # KL divergence
    kl_div_baseline = KLdivergence(norm_samp, data_samp)
    kl_div_pred = KLdivergence(pred_samp, data_samp)
    results["kl_div_normal_baseline"] = np.float64(kl_div_baseline)
    results["kl_div_predictive"] = np.float64(kl_div_pred)

    # Difference in mean
    mean_diff_baseline = difference_mean(norm_samp, data_samp)
    mean_diff_pred = difference_mean(pred_samp, data_samp)
    results["mean_diff_normal_baseline"] = np.float64(mean_diff_baseline)
    results["mean_diff_predictive"] = np.float64(mean_diff_pred)

    # Difference in variance
    var_diff_baseline = difference_var(norm_samp, data_samp)
    var_diff_pred = difference_var(pred_samp, data_samp)
    results["var_diff_normal_baseline"] = np.float64(var_diff_baseline)
    results["var_diff_predictive"] = np.float64(var_diff_pred)

    # Difference in standard deviation
    std_diff_baseline = difference_std(norm_samp, data_samp)
    std_diff_pred = difference_std(pred_samp, data_samp)
    results["std_diff_normal_baseline"] = np.float64(std_diff_baseline)
    results["std_diff_predictive"] = np.float64(std_diff_pred)

    # Save results
    with open(f"{DIR}/{NAME}/results/results.json", "w") as f:
        json.dump(results, f)