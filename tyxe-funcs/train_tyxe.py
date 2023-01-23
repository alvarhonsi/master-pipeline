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
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.infer import SVI, MCMC, NUTS, HMC, Trace_ELBO, Predictive
import tyxe
from datetime import timedelta
import dill
import time
import json
import argparse

def train(config, DIR):

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
    LR = config.getfloat("LR")
    TRAIN_BATCH_SIZE = config.getint("TRAIN_BATCH_SIZE")

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(f"{DIR}/data/{NAME}")

    # Ready model directory
    if not os.path.exists(f"{DIR}/{MODEL_PATH}"):
        os.mkdir(f"{DIR}/{MODEL_PATH}")

    if not os.path.exists(f"{DIR}/{MODEL_PATH}/{NAME}"):
        os.mkdir(f"{DIR}/{MODEL_PATH}/{NAME}")

    # Check if GPU is available
    if DEVICE[:4] == "cuda" and not torch.cuda.is_available():
        raise ValueError("GPU not available")
    elif DEVICE[:4] == "cuda" and torch.cuda.is_available():
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("Cuda enabled !")

    # Create datasets and dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    test_dataset = TensorDataset(x_test, y_test)

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=2, shuffle=True)
    ood_dataloader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=2, shuffle=True)

    # Create model
    try:
        model = model_types[MODEL_TYPE]
    except KeyError:
        raise ValueError(f"Model type {MODEL_TYPE} not supported. Supported types: {model_types.keys()}")
    net = model(X_DIM, Y_DIM, hidden_features=HIDDEN_FEATURES, device=DEVICE)

    print("Model device: ", net.device)

    # Create inference model
    if INFERENCE_TYPE == "svi":
        prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE)))
        likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=len(train_dataloader.sampler))
        guide = tyxe.guides.AutoNormal
        bnn = tyxe.VariationalBNN(net, prior, likelihood, guide)
    elif INFERENCE_TYPE == "mcmc":
        prior = tyxe.priors.IIDPrior(dist.Normal(torch.zeros(1, device=DEVICE), torch.ones(1, device=DEVICE)))
        likelihood = tyxe.likelihoods.HomoskedasticGaussian(scale=0.1, dataset_size=len(train_dataloader.sampler))
        kernel = pyro.infer.mcmc.NUTS
        bnn = tyxe.MCMC_BNN(net, prior, likelihood, kernel)
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

    if INFERENCE_TYPE == "svi":
        start = time.time()
        def callback(b, i, avg_elbo):
            avg_err, avg_ll = 0., 0.
            b.eval()
            for x, y in iter(val_dataloader):
                err, ll = b.evaluate(x.to(DEVICE), y.to(DEVICE), num_predictions=100)
                avg_err += err / len(val_dataloader.sampler)
                avg_ll += ll / len(val_dataloader.sampler)
            
            td = timedelta(seconds=round(time.time() - start))
            print(f"[{td}] ELBO={avg_elbo}; test error={100 * avg_err:.2f}%; LL={avg_ll:.4f}")
            b.train()
        
        optim = pyro.optim.Adam({"lr": LR})
        bnn.fit(train_dataloader, optim, num_epochs=EPOCHS, callback=callback, device=DEVICE)
    elif INFERENCE_TYPE == "mcmc":
        bnn.fit(train_dataloader, num_samples=MCMC_NUM_SAMPLES, device=DEVICE)
    else:
        raise ValueError(f"Inference type {INFERENCE_TYPE} not supported. Supported types: svi, mcmc")

    print("bnn", bnn.state_dict())
    print("bnn net", bnn.net.state_dict())
    print("bnn net_guide", bnn.net_guide.state_dict())

    # Save model
    if SAVE_MODEL:
        if INFERENCE_TYPE == "svi":
            pyro.get_param_store().save(f"{DIR}/{MODEL_PATH}/{NAME}/svi_param_store.pt")
            torch.save({"net" : bnn.net.state_dict(), "bnn": bnn.state_dict()}, f"{DIR}/{MODEL_PATH}/{NAME}/checkpoint.pt")
            print(f"Saved model and params to {DIR}/{MODEL_PATH}/{NAME}")
        elif INFERENCE_TYPE == "mcmc":
            mcmc = bnn._mcmc

            torch.save(bnn.state_dict(), f"{DIR}/{MODEL_PATH}/{NAME}/mcmc_state_dict.pt")
            with open(f"{DIR}/{MODEL_PATH}/{NAME}/mcmc.pkl", 'wb') as f:
                dill.dump(mcmc, f)
            #torch.save(mcmc, f"{DIR}/{MODEL_PATH}/{NAME}/mcmc.pt")
            print(f"Saved model and samples to {DIR}/{MODEL_PATH}/{NAME}")

    return bnn

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


    train(config, DIR)