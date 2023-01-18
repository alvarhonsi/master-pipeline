from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import KL_div_knn_estimation, KLdivergence
from modules.distributions import PredictivePosterior, DataDistribution, NormalPosterior
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.infer import Predictive, NUTS
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Normal
import os
import argparse

def main():
    parser = argparse.ArgumentParser(
                    prog = 'Generate datasets',
                    description = 'Generates datasets for training, testing and validation based on a given function and noise level. Configurations are read from config.ini. The generated datasets are saved in a named data directory.',
                    epilog = 'Example: python generate.py -c DEFAULT')
    parser.add_argument('-c', '--config', help='Name of configuration section', default="DEFAULT")
    args = parser.parse_args()

    # Load config
    config = read_config("config.ini")
    config = config[args.config]

    NAME = config["NAME"]
    SEED = config.getint("SEED")
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")

    DATA_FUNC = config["DATA_FUNC"]
    MU = config.getint("MU")
    SIGMA = config.getint("SIGMA")

    MODEL_TYPE = config["MODEL_TYPE"]
    HIDDEN_FEATURES = config.getlist("HIDDEN_FEATURES")
    MODEL_PATH = config["MODEL_PATH"]

    INFERENCE_TYPE = config["INFERENCE_TYPE"]
    SVI_GUIDE = config["SVI_GUIDE"]
    SVI_ELBO = config["SVI_ELBO"]
    MCMC_KERNEL = config["MCMC_KERNEL"]
    MCMC_NUM_SAMPLES = config.getint("MCMC_NUM_SAMPLES")
    MCMC_NUM_WARMUP = config.getint("MCMC_NUM_WARMUP")
    MCMC_NUM_CHAINS = config.getint("MCMC_NUM_CHAINS")

    RESULTS_PATH = config["RESULTS_PATH"]
    NUM_X_SAMPLES = config.getint("NUM_X_SAMPLES")
    NUM_DIST_SAMPLES = config.getint("NUM_DIST_SAMPLES")
    EVAL_BATCH_SIZE = config.getint("EVAL_BATCH_SIZE")
    

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pyro.clear_param_store()

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(NAME)

    # Make dataset
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=4)

    # Ready results directory
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    if not os.path.exists(f"{RESULTS_PATH}/{NAME}"):
        os.mkdir(f"{RESULTS_PATH}/{NAME}")
    

    # Load model
    BNN = model_types[MODEL_TYPE]
    model = BNN(X_DIM, Y_DIM, HIDDEN_FEATURES)

    if INFERENCE_TYPE == "svi":
        guide = AutoDiagonalNormal(model)
        optim = pyro.optim.Adam({"lr": 1e-3})
        inference_model = SVIInferenceModel(model, guide, optim)
    elif INFERENCE_TYPE == "mcmc":
        mcmc_kernel = NUTS(model)
        inference_model = MCMCInferenceModel(model, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS)
    else:
        raise ValueError(f"Invalid inference type: {INFERENCE_TYPE}")

    inference_model.load(f"{MODEL_PATH}/{NAME}")

    # TODO: change to use full data loader
    x, y = next(iter(val_dataloader))

    idx = np.random.choice(range(len(x)), NUM_X_SAMPLES)
    sample_x = x[idx]
    print("sample x:", sample_x.shape)

    #Baseline normal distribution
    base_dist = NormalPosterior(0, 1, sample_x)
    normal_samples = base_dist.sample(NUM_DIST_SAMPLES)

    #Sample true distribution from data
    func = data_functions[DATA_FUNC]
    data_dist = DataDistribution(func, MU, SIGMA, sample_x)
    data_samp = data_dist.sample(NUM_DIST_SAMPLES)

    #Sample posterior distribution from model
    pred_samp = inference_model.predict(sample_x, NUM_DIST_SAMPLES)

    """
    # Baseline kl divergence (normal)
    kl_div_norm = KLdivergence(normal_samples, data_samp)
    print(f"KL divergence baseline: {kl_div_norm}")
    kl_div_norm_knn = KL_div_knn_estimation(normal_samples, data_samp)
    print(f"KL divergence baseline (knn): {kl_div_norm_knn}")

    #Compare true and posterior distributions
    kl_div = KLdivergence(pred_samp, data_samp)
    print(f"KL divergence: {kl_div}")
    kl_div_knn = KL_div_knn_estimation(pred_samp, data_samp)
    print(f"KL divergence (knn): {kl_div_knn}")
    """
    #plot_comparison(pred_samp[:, 0], data_samp[:, 0], f"{RESULT_PATH}/{name}/comparison.png")
    plot_comparison_grid(data_samp, pred_samp, grid_size=2, save_path=f"{RESULTS_PATH}/{NAME}/comparison_grid.png")



if __name__ == "__main__":
    main()