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
    IN_DOMAIN_SAMPLE_SPACE = dataset_config.gettuple("IN_DOMAIN_SAMPLE_SPACE")
    OUT_DOMAIN_SAMPLE_SPACE = dataset_config.gettuple("OUT_DOMAIN_SAMPLE_SPACE")
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

    train = generate_dataset((TRAIN_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=IN_DOMAIN_SAMPLE_SPACE)
    val = generate_dataset((VAL_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=IN_DOMAIN_SAMPLE_SPACE)

    test = generate_dataset((TEST_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=OUT_DOMAIN_SAMPLE_SPACE)
    test_in_domain = generate_dataset((TEST_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=IN_DOMAIN_SAMPLE_SPACE)
    test_out_domain = generate_dataset((TEST_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=OUT_DOMAIN_SAMPLE_SPACE, void_space=IN_DOMAIN_SAMPLE_SPACE)

    train_complete = np.hstack((train[0], train[1].reshape(-1, 1)))
    val_complete = np.hstack((val[0], val[1].reshape(-1, 1)))

    test_complete = np.hstack((test[0], test[1].reshape(-1, 1)))
    test_in_domain_complete = np.hstack((test_in_domain[0], test_in_domain[1].reshape(-1, 1)))
    test_out_domain_complete = np.hstack((test_out_domain[0], test_out_domain[1].reshape(-1, 1)))

    # Save datasets
    np.savetxt(f"{DIR}/{NAME}/train.csv", train_complete, delimiter=",") #detach()? numpy()?
    np.savetxt(f"{DIR}/{NAME}/val.csv", val_complete, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/test.csv", test_complete, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/test_in_domain.csv", test_in_domain_complete, delimiter=",")
    np.savetxt(f"{DIR}/{NAME}/test_out_domain.csv", test_out_domain_complete, delimiter=",")

    # Visualize datasets
    if SIGMA == 0:
        return
    
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