from modules.datageneration import generate_dataset, data_functions
from modules.config import read_config
from modules.distributions import DataDistribution  
from modules.plots import plot_distribution
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import json
import os
import argparse

def gen(dataset_config, DIR):

    NAME = dataset_config["NAME"]
    SEED = dataset_config.getint("SEED")
    X_DIM = dataset_config.getint("X_DIM")
    Y_DIM = dataset_config.getint("Y_DIM")

    TRAIN_SIZE = dataset_config.getint("TRAIN_SIZE")
    TEST_SIZE = dataset_config.getint("TEST_SIZE")
    VAL_SIZE = dataset_config.getint("VAL_SIZE")
    TRAIN_SAMPLE_SPACE = dataset_config.gettuple("TRAIN_SAMPLE_SPACE")
    TEST_SAMPLE_SPACE = dataset_config.gettuple("TEST_SAMPLE_SPACE")
    FUNC = dataset_config["DATA_FUNC"]
    MU = dataset_config.getint("MU")
    SIGMA = dataset_config.getint("SIGMA")
        

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    #ready data directory
    if not os.path.exists(f"{DIR}/{NAME}"):
        os.mkdir(f"{DIR}/{NAME}")

    # Generate datasets
    print(f"====== Generating profile {NAME} ======")

    train = generate_dataset((TRAIN_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TRAIN_SAMPLE_SPACE)
    val = generate_dataset((VAL_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TRAIN_SAMPLE_SPACE)
    test = generate_dataset((TEST_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TEST_SAMPLE_SPACE)

    train_complete = np.hstack((train[0], train[1].reshape(-1, 1)))
    val_complete = np.hstack((val[0], val[1].reshape(-1, 1)))
    test_complete = np.hstack((test[0], test[1].reshape(-1, 1)))

    # Save datasets
    np.savetxt(f"{DIR}/{NAME}/train.csv", train_complete, delimiter=",") #detach()? numpy()?
    np.savetxt(f"{DIR}/{NAME}/val.csv", val_complete, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/test.csv", test_complete, delimiter=",")

    # Visualize datasets
    sample_train_x = train[0][[0]]
    func = data_functions[FUNC]
    sample_train_y = func(sample_train_x)
    sample_train_x = torch.tensor(sample_train_x)
    train_sample_dist = DataDistribution(func, MU, SIGMA, sample_train_x)
    train_sample_posterior = train_sample_dist.sample(100000).squeeze()

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_distribution(train_sample_posterior, ax=ax)
    ax.axvline(sample_train_y, color="red", label=f"f(x) = {sample_train_y.item():.2f}")
    ax.legend()
    ax.set_title(f"Posterior distribution of y for sample x = {[round(x.item(), 2) for x in sample_train_x[0]]}")
    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    plt.savefig(f"{DIR}/{NAME}/sample_posterior.png")


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
    config = read_config("config.ini")
    config = config[args.config]

    gen(config, DIR)