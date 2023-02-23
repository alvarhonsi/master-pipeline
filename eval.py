from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import KLdivergence, difference_mean, difference_std, difference_var, KL_div_knn_estimation
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

def evaluate_posterior(posterior_samples, data_samples):
    results = {}
    # KL divergence
    kl_div_data = KL_div_knn_estimation(data_samples, data_samples)
    kl_div = KL_div_knn_estimation(posterior_samples, data_samples)
    #print("KL divergence: ", kl_div)
    print("KL divergence2: ", kl_div)
    print("KL divergence data: ", kl_div_data)
    results["kl_div"] = np.float64(kl_div)

    # Difference in mean
    mean_diff = difference_mean(posterior_samples, data_samples)
    results["mean_diff"] = np.float64(mean_diff)

    # Difference in variance
    var_diff = difference_var(posterior_samples, data_samples)
    results["var_diff"] = np.float64(var_diff)

    # Difference in standard deviation
    std_diff = difference_std(posterior_samples, data_samples)
    results["std_diff"] = np.float64(std_diff)

    return results

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
    _, (x_val, y_val), (x_test, y_test), (x_test_in_domain, y_test_in_domain), (x_test_out_domain, y_test_out_domain) = load_data(f"{DIR}/datasets/{DATASET}", load_train=False)

    # Make dataset
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    test_in_domain_dataset = TensorDataset(x_test_in_domain, y_test_in_domain)
    test_out_domain_dataset = TensorDataset(x_test_out_domain, y_test_out_domain)

    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_in_domain_dataloader = DataLoader(test_in_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    test_out_domain_dataloader = DataLoader(test_out_domain_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=3)
    
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
    
    # Ready samples directory
    if not os.path.exists(f"{DIR}/{NAME}/results/samples"):
        os.mkdir(f"{DIR}/{NAME}/results/samples")

    # samp_x: (NUM_X_SAMPLES, X_DIM), samp_y: (NUM_X_SAMPLES)
    test_x_sample, _ = draw_data_samples(test_dataloader, NUM_X_SAMPLES)
    test_in_domain_x_sample, _ = draw_data_samples(test_in_domain_dataloader, NUM_X_SAMPLES)
    test_out_domain_x_sample, _ = draw_data_samples(test_out_domain_dataloader, NUM_X_SAMPLES)
    

    #Baseline normal distribution
    # normal_samples: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    base_dist = NormalPosterior(0, 1, test_x_sample)
    base_in_domain_dist = NormalPosterior(0, 1, test_in_domain_x_sample)
    base_out_domain_dist = NormalPosterior(0, 1, test_out_domain_x_sample)

    normal_samples = base_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    normal_in_domain_samples = base_in_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    normal_out_domain_samples = base_out_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.savetxt(f"{DIR}/{NAME}/results/samples/baseline_samples.csv", normal_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/baseline_in_domain_samples.csv", normal_in_domain_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/baseline_out_domain_samples.csv", normal_out_domain_samples, delimiter=",")
    
    #Sample true distribution from data
    # data_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    func = data_functions[DATA_FUNC]
    data_dist = DataDistribution(func, MU, SIGMA, test_x_sample)
    data_in_domain_dist = DataDistribution(func, MU, SIGMA, test_in_domain_x_sample)
    data_out_domain_dist = DataDistribution(func, MU, SIGMA, test_out_domain_x_sample)

    data_samples = data_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_in_domain_samples = data_in_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()
    data_out_domain_samples = data_out_domain_dist.sample(NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.savetxt(f"{DIR}/{NAME}/results/samples/data_samples.csv", data_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/data_in_domain_samples.csv", data_in_domain_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/data_out_domain_samples.csv", data_out_domain_samples, delimiter=",")
    

    #Sample posterior distribution from model
    # pred_samp: (NUM_DIST_SAMPLES, NUM_X_SAMPLES)
    pred_samples = inference_model.predict(test_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
    pred_in_domain_samples = inference_model.predict(test_in_domain_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()
    pred_out_domain_samples = inference_model.predict(test_out_domain_x_sample, NUM_DIST_SAMPLES).cpu().detach().numpy()

    np.savetxt(f"{DIR}/{NAME}/results/samples/predictive_samples.csv", pred_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/predictive_in_domain_samples.csv", pred_in_domain_samples, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/results/samples/predictive_out_domain_samples.csv", pred_out_domain_samples, delimiter=",")
    

    # Evaluate

    results = {}

    #test_rmse = inference_model.evaluate(test_dataloader)
    #test_in_domain_rmse = inference_model.evaluate(test_in_domain_dataloader)
    #test_out_domain_rmse = inference_model.evaluate(test_out_domain_dataloader)
    #results["test_rmse"] = np.float64(test_rmse)
    #results["test_in_domain_rmse"] = np.float64(test_in_domain_rmse)
    #results["test_out_domain_rmse"] = np.float64(test_out_domain_rmse)

    results["predictive"] = evaluate_posterior(pred_samples, data_samples)
    results["predictive_in_domain"] = evaluate_posterior(pred_in_domain_samples, data_in_domain_samples)
    results["predictive_out_domain"] = evaluate_posterior(pred_out_domain_samples, data_out_domain_samples)

    # Save results
    with open(f"{DIR}/{NAME}/results/results.json", "w") as f:
        json.dump(results, f, indent=4)