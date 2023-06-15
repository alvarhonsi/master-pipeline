from modules.config import read_config
from modules.datageneration import load_data, data_functions
from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.plots import plot_comparison, plot_comparison_grid
from modules.metrics import KL_div_knn_estimation, KLdivergence
from modules.distributions import PredictivePosterior, DataDistribution, NormalPosterior
from modules.context import set_default_tensor_type
import tyxe
import dill
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import pyro.distributions as dist
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

def evaluate_bnn(bnn, dataloader, num_predictions=1, device="cpu"):
    avg_err, avg_ll = 0., 0.
    bnn.eval()
    for x, y in iter(dataloader):
        err, ll = bnn.evaluate(x.to(device), y.to(device), num_predictions=num_predictions)
        avg_err += err / len(dataloader.sampler)
        avg_ll += ll / len(dataloader.sampler)

    return avg_err, avg_ll

def eval(config, DIR, bnn=None):

    NAME = config["NAME"]
    SEED = config.getint("SEED")
    DEVICE = config["DEVICE"]
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

    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")
    elif DEVICE[:4] == "cuda" and torch.cuda.is_available():
        print("Cuda enabled !")
        
    

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    pyro.clear_param_store()

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(f"{DIR}/data/{NAME}")

    # Make dataset
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)
    print(f"Test dataset size: {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=EVAL_BATCH_SIZE)
    val_dataloader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE)

    print("Dataloader device: ", next(iter(train_dataloader))[0].device)
    # Ready results directory
    if not os.path.exists(f"{DIR}/{RESULTS_PATH}"):
        os.mkdir(f"{DIR}/{RESULTS_PATH}")

    if not os.path.exists(f"{DIR}/{RESULTS_PATH}/{NAME}"):
        os.mkdir(f"{DIR}/{RESULTS_PATH}/{NAME}")
    

    # Load model
    if bnn is None:
        model = model_types[MODEL_TYPE]
        net = model(X_DIM, Y_DIM, HIDDEN_FEATURES, device=DEVICE)

        prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE)))
        likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=len(train_dataloader.sampler))

        if INFERENCE_TYPE == "svi":
            guide = tyxe.guides.AutoNormal
            bnn = tyxe.VariationalBNN(net, prior, likelihood, guide)

            pyro.get_param_store().load(f"{DIR}/{MODEL_PATH}/{NAME}/svi_param_store.pt")
            checkpoint = torch.load(f"{DIR}/{MODEL_PATH}/{NAME}/checkpoint.pt", map_location=DEVICE)
            bnn.net.load_state_dict(checkpoint["net"])
            print(bnn.state_dict())
            sd = bnn.state_dict()
            sd = checkpoint["bnn"]
            print(bnn.state_dict())
            
        elif INFERENCE_TYPE == "mcmc":
            kernel = pyro.infer.mcmc.NUTS
            bnn = tyxe.MCMC_BNN(net, prior, likelihood, kernel)

            with open(f"{DIR}/{MODEL_PATH}/{NAME}/mcmc.pkl", 'rb') as f:
                mcmc = dill.load(f)

            bnn.load_state_dict(torch.load(f"{DIR}/{MODEL_PATH}/{NAME}/mcmc_state_dict.pt", map_location=DEVICE))
            bnn._mcmc = mcmc

            print(f"Saved model and samples to {DIR}/{MODEL_PATH}/{NAME}")
        else:
            raise ValueError(f"Invalid inference type: {INFERENCE_TYPE}")

    print(f"======Evaluating model with device: {DEVICE}======")

    # Model RMSE
    #train_rmse = inference_model.evaluate(train_dataloader)
    #val_rmse = inference_model.evaluate(val_dataloader)
    print("Evaluating model...")
    val_error, val_ll = evaluate_bnn(bnn, test_dataloader, num_predictions=100, device=DEVICE)
    print(f"Validation error: {val_error}")
    print(f"Validation log-likelihood: {val_ll}")
    test_error, test_ll = evaluate_bnn(bnn, test_dataloader, num_predictions=100, device=DEVICE)
    print(f"Test error: {test_error}")
    print(f"Test log-likelihood: {test_ll}")

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
    print("Sampling from model!!!!")
    pred_samp = bnn.predict(samp_x, NUM_DIST_SAMPLES, aggregate=False)
    print(pred_samp)
    print(pred_samp.shape)
    


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
    with open(f"{DIR}/{RESULTS_PATH}/{NAME}/results.txt", "w") as f:
        f.write(f"Validation error: {val_error}\n")
        f.write(f"Validation log-likelihood: {val_ll}\n")
        f.write(f"Test error: {test_error}\n")
        f.write(f"Test log-likelihood: {test_ll}\n")
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