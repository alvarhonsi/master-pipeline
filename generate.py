from modules.datageneration import generate_dataset, data_functions
from modules.config import read_config
from modules.distributions import DataDistribution  
from modules.plots import plot_distribution
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pyro
import json
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

    DATA_PATH = config["DATA_PATH"]
    TRAIN_SIZE = config.getint("TRAIN_SIZE")
    TEST_SIZE = config.getint("TEST_SIZE")
    VAL_SIZE = config.getint("VAL_SIZE")
    TRAIN_SAMPLE_SPACE = config.gettuple("TRAIN_SAMPLE_SPACE")
    TEST_SAMPLE_SPACE = config.gettuple("TEST_SAMPLE_SPACE")
    FUNC = config["DATA_FUNC"]
    MU = config.getint("MU")
    SIGMA = config.getint("SIGMA")
        

    # Reproducibility
    pyro.set_rng_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    #ready data directory
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)

    if not os.path.exists(f"{DATA_PATH}/{NAME}"):
        os.mkdir(f"{DATA_PATH}/{NAME}")

    # Generate datasets
    print('''Generating datasets with the following parameters: 
X_dim: {}, Y_dim: {}
Train size: {}, Test size: {}, Val size: {}
Train sample space: {}, Test sample space: {}
Data function: {}, Mu: {}, Sigma: {}
-------------------------------------------------
'''.format(X_DIM, Y_DIM, TRAIN_SIZE, TEST_SIZE, VAL_SIZE, TRAIN_SAMPLE_SPACE, TEST_SAMPLE_SPACE, FUNC, MU, SIGMA))

    train = generate_dataset((TRAIN_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TRAIN_SAMPLE_SPACE)
    val = generate_dataset((VAL_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TRAIN_SAMPLE_SPACE)
    test = generate_dataset((TEST_SIZE, X_DIM), FUNC, MU, SIGMA, sample_space=TEST_SAMPLE_SPACE)

    train_df = pd.DataFrame(train[0], columns=[f"x{i+1}" for i in range(X_DIM)])
    train_df["y"] = train[1]

    val_df = pd.DataFrame(val[0], columns=[f"x{i+1}" for i in range(X_DIM)])
    val_df["y"] = val[1]

    test_df = pd.DataFrame(test[0], columns=[f"x{i+1}" for i in range(X_DIM)])
    test_df["y"] = test[1]

    # Save datasets
    train_df.to_csv(f"{DATA_PATH}/{NAME}/train.csv", index=False)
    val_df.to_csv(f"{DATA_PATH}/{NAME}/val.csv", index=False)
    test_df.to_csv(f"{DATA_PATH}/{NAME}/test.csv", index=False)

    # Visualize datasets
    sample_train_x = train[0][[0]]
    func = data_functions[FUNC]
    sample_train_y = func(sample_train_x)
    sample_train_x = torch.tensor(sample_train_x)
    train_sample_dist = DataDistribution(func, MU, SIGMA, sample_train_x)
    train_sample_posterior = train_sample_dist.sample(1000000)[0]

    fig, ax = plt.subplots(figsize=(15, 10))
    plot_distribution(train_sample_posterior, ax=ax)
    ax.axvline(sample_train_y, color="red", label=f"f(x) = {sample_train_y.item():.2f}")
    ax.legend()
    ax.set_title(f"Posterior distribution of y for sample x = {[round(x.item(), 2) for x in sample_train_x[0]]}")
    ax.set_xlabel("y")
    ax.set_ylabel("Density")
    plt.savefig(f"{DATA_PATH}/{NAME}/train_sample_posterior.png")


if __name__ == "__main__":
    main()