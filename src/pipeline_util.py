import torch
import os
import pyro
import dill
import numpy as np


def draw_data_samples(dataloader, num_samples=10, device="cpu"):
    xs, ys = [], []
    for x, y in dataloader:
        xs.append(x)
        ys.append(y)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)

    idx = np.random.choice(range(len(x)), num_samples)
    sample_x = x[idx]
    sample_y = y[idx]
    return sample_x.to(device), sample_y.to(device)


def save_bnn(bnn, config, save_path=None):
    if save_path == None:
        raise Exception("No save path specified")

    inference_type = config["INFERENCE_TYPE"]

    if inference_type == "svi":
        pyro.get_param_store().save(save_path)
        print("Saved SVI model to", save_path)
        file_stats = os.stat(save_path)
        print(
            f'File Size is {file_stats.st_size / (1024 * 1024)} MB')

    elif inference_type == "mcmc":
        torch.save({"samples": bnn._mcmc._samples},
                   save_path, pickle_module=dill)
        print("Saved MCMC samples to", save_path)
        file_stats = os.stat(save_path)
        print(
            f'File Size is {file_stats.st_size / (1024 * 1024)} MB')

    elif inference_type == "nn":
        torch.save(bnn.net.state_dict(), save_path)
        print("Saved Baseline_nn model to", save_path)
        file_stats = os.stat(save_path)
        print(
            f'File Size is {file_stats.st_size / (1024 * 1024)} MB')

    else:
        raise Exception("Unknown inference type")


def load_bnn(bnn, config, load_path=None, device=None):
    if load_path == None:
        raise Exception("No load path specified")

    inference_type = config["INFERENCE_TYPE"]
    input_dim = config.getint("X_DIM")

    if inference_type == "svi":
        pyro.get_param_store().load(load_path, map_location=device)

        # Run a prediction to initialize the model using dummy data
        dummy_x = torch.zeros(1, input_dim).to(device)
        bnn.predict(dummy_x)
        print("Loaded SVI model from", load_path)

    elif inference_type == "mcmc":
        checkpoint = torch.load(
            load_path, map_location=device, pickle_module=dill)
        ## init mcmc
        dummy_x, dummy_y = torch.zeros(1, input_dim).to(device), torch.zeros(1, 1).to(device)
        bnn.fit(dummy_x, dummy_y)

        bnn._mcmc._samples = checkpoint["samples"]
        print("Loaded MCMC model from", load_path)

    elif inference_type == "nn":
        bnn.net.load_state_dict(torch.load(
            load_path, map_location=device))
        print("Loaded Baseline_nn model from", load_path)

    else:
        raise Exception("Unknown inference type")

    return bnn
