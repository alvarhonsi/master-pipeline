from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import KL_div_knn_estimation, KLdivergence
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

    # Ready data directory
    if not os.path.exists(f"{DIR}/{NAME}"):
        os.mkdir(f"{DIR}/{NAME}")

    print("Loading data...")
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(f"{DIR}/datasets/{DATASET}")

    print("Preparing datasets...")
    # Make dataset
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    print("Preparing dataloaders...")

    train_dataloader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    # Ready results directory
    if not os.path.exists(f"{DIR}/{NAME}/results"):
        os.mkdir(f"{DIR}/{NAME}/results")
    
    # Load model
    print("Loading model...")
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

    print(f"using device: {DEVICE}")
    print(f"====== evaluating profile {NAME} ======")

    # Model RMSE
    #train_rmse = inference_model.evaluate(train_dataloader)
    #val_rmse = inference_model.evaluate(val_dataloader)
    print("Evaluating model...")
    test_rmse = inference_model.evaluate(test_dataloader)

    # Draw data samples
    # samp_x: (NUM_X_SAMPLES, X_DIM), samp_y: (NUM_X_SAMPLES)
    samp_x, samp_y = draw_data_samples(test_dataloader, NUM_X_SAMPLES)
    

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

    print("Normal samples shape: ", norm_samp.shape)
    print("Data samples shape: ", data_samp.shape)
    print("Predictive samples shape: ", pred_samp.shape)
    # KL divergence
    kl_div_baseline = KLdivergence(norm_samp, data_samp)
    print(f"KL divergence baseline: {kl_div_baseline}")
    kl_div_pred = KLdivergence(pred_samp, data_samp)
    print(f"KL divergence predictive dist: {kl_div_pred}")

    # Save results
    with open(f"{DIR}/{NAME}/results/results.txt", "w") as f:
        f.write(f"Test RMSE: {test_rmse}\n")
        f.write(f"KL divergence baseline: {kl_div_baseline}\n")
        f.write(f"KL divergence predictive dist: {kl_div_pred}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Generate datasets',
                    description = 'Generates datasets for training, testing and validation based on a given function and noise level. Configurations are read from config.ini. The generated datasets are saved in a named data directory.',
                    epilog = 'Example: python generate.py -c DEFAULT')
    parser.add_argument('-c', '--config', help='Name of configuration section', default="DEFAULT")
    parser.add_argument('-dir', '--directory', help='Name of base directory where data will be stored', default=".")
    args = parser.parse_args()

    # Set base directory
    DIR = args.directory

    # Load config
    config = read_config(f"{DIR}/config.ini")
    config = config[args.config]

    eval(config, DIR)