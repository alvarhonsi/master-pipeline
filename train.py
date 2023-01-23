from modules.models import model_types
from modules.inference import MCMCInferenceModel, SVIInferenceModel
from modules.datageneration import load_data
from modules.config import read_config
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, MCMC, NUTS, HMC, Trace_ELBO, Predictive
from datetime import timedelta
import time
import json
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
    DEVICE = config["DEVICE"]
    SEED = config.getint("SEED")
    X_DIM = config.getint("X_DIM")
    Y_DIM = config.getint("Y_DIM")

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
    MODEL_PATH = config["MODEL_PATH"]
    EPOCHS = config.getint("EPOCHS")
    TRAIN_BATCH_SIZE = config.getint("TRAIN_BATCH_SIZE")
    
    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")
    elif DEVICE[:4] == "cuda" and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("Cuda enabled !")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    (x_train, y_train), (x_val, y_val), _ = load_data(NAME)

    # Ready model directory
    if not os.path.exists(MODEL_PATH):
        os.mkdir(MODEL_PATH)

    if not os.path.exists(f"{MODEL_PATH}/{NAME}"):
        os.mkdir(f"{MODEL_PATH}/{NAME}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=3,
    generator=torch.Generator(device=DEVICE))
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=3,
    generator=torch.Generator(device=DEVICE))

    # Create model
    try:
        BNN = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    model = BNN(X_DIM, Y_DIM, hidden_features=HIDDEN_FEATURES, device=DEVICE)

    # Create inference model
    if INFERENCE_TYPE == "svi":
        guide = AutoDiagonalNormal(model)
        optim = pyro.optim.Adam({"lr": 1e-3})
        inference_model = SVIInferenceModel(model, guide, optim, EPOCHS, device=DEVICE)
    elif INFERENCE_TYPE == "mcmc":
        mcmc_kernel = NUTS(model, adapt_step_size=True, max_tree_depth=10)
        #mcmc_kernel = HMC(model, adapt_step_size=True)
        inference_model = MCMCInferenceModel(model, mcmc_kernel, num_samples=MCMC_NUM_SAMPLES, 
        num_warmup=MCMC_NUM_WARMUP, num_chains=MCMC_NUM_CHAINS, device=DEVICE)
    else:
        raise ValueError(f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")



    # RUN TRAINING
    print('Training model type {} with inference method {}: \n'.format(MODEL_TYPE, INFERENCE_TYPE)
        + 'Model class {} with architecture: \n'.format(model.__class__.__name__)
        + 'Input: {}, Output: {}, Hidden: {} \n'.format(X_DIM, Y_DIM, HIDDEN_FEATURES)
        + 'Train set size: {}, Validation set size: {} \n'.format(x_train.shape[0], x_val.shape[0])
        + 'Training for {} epochs with batch size {} \n'.format(EPOCHS, TRAIN_BATCH_SIZE)
    )
    print('====== Training on device: {} ======\n'.format(DEVICE))
    train_stats = inference_model.fit(train_dataloader)
    if SAVE_MODEL:
        inference_model.save(f"{MODEL_PATH}/{NAME}")
        with open(f"{MODEL_PATH}/{NAME}/train_stats.json", "w") as json_file:
            json.dump(train_stats, json_file)

if __name__ == "__main__":
    main()