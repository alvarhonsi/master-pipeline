import numpy as np
from numpy import genfromtxt
import torch
import json
import scipy

def identity(x, noise=0) -> np.array:
    return x + noise

def linear_func(x, noise=0) -> np.array:
    return (2*x + 1) + noise

def polynomial_func(x, noise=0) -> np.array:
    return (0.5*x**2 - 3*x + 1) + noise

def sinusoidal_func(xs, noise=0) -> np.array:
    return np.squeeze(np.sin(xs.sum(axis=1)) * 2 + 3) + noise

def sinusoidal_combination(x, noise=0) -> np.array:
    if len(x.shape) == 2 and x.shape[1] > 1:
        raise Exception("Function only takes 1D input")
    if len(x.shape) == 2:
        x = x.reshape(-1)

    return np.squeeze(x + 0.3 * np.sin(2 * (x + noise*0.6)) + 0.3 * np.sin(4 * (x + noise*0.4)) + noise) 

def normal_percentile(mu, sigma, percentile) -> float:
    return mu + sigma * scipy.stats.norm.ppf(percentile)

def tendim_sinusoidal_combination(xs, noise=0) -> np.array:
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3], xs[:, 4], xs[:, 5], xs[:, 6], xs[:, 7], xs[:, 8], xs[:, 9]

    return 0*x1 + np.sin(x2 + x3) + np.sin(x4 * x5) + np.sin(x6 + x7) + np.sin(x8 * x9) + 0*x10 + noise

def tendim_sinusoidal_weighted(xs, noise=0) -> np.array:
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = xs[:, 0], xs[:, 1], xs[:, 2], xs[:, 3], xs[:, 4], xs[:, 5], xs[:, 6], xs[:, 7], xs[:, 8], xs[:, 9]

    return np.sin(x1) * 6*np.sin(x2) + 8 * np.sin(x3) + 4 * np.sin(x4) * np.sin(x5) + 6*np.sin(x6) + 10 * np.sin(x7) * np.sin(x8) + 5 * np.sin(x9) * np.sin(x10) + noise

def sum(xs, noise=0) -> np.array:
    return np.sum(xs, axis=1) + noise

'''
    Dictionary of functions that can be used in data_gen
    functions should take a numpy array [n, x_dim] and return a numpy array [n] - the output of the function
'''
data_functions = {
    "identity": identity,
    "linear": linear_func,
    "polynomial": polynomial_func,
    "sinusoidal": sinusoidal_func,
    "sinusoidal_combination": sinusoidal_combination,
    "sum": sum,
    "tendim_sinusoidal_combination": tendim_sinusoidal_combination,
    "tendim_sinusoidal_weighted": tendim_sinusoidal_weighted
} 
    
'''
    Generates a dataset of size sample_size with the given function and noise
    The function is expected to be a function that takes a numpy array and returns a numpy array
    The noise is added as a normal distribution with mean mu and standard deviation sigma

    sample_size: tuple of ints (sample_size, x_dim)
    func: string - name of function to use
    mu: float - mean of noise
    sigma: float - standard deviation of noise
    sample_space: tuple of floats (lower, upper) - range of x values

    returns: tuple of numpy arrays (x, y)
'''
def data_gen(sample_size, func, mu, sigma, sample_space, void_space=None):
    try:
        func = data_functions[func]
    except KeyError:
        raise Exception("Invalid function name")

    lower, upper = sample_space

    if void_space is not None:
        lower_stop, upper_start = void_space
        x1 = np.random.uniform(lower, lower_stop, (sample_size[0]//2, sample_size[1]))
        x2 = np.random.uniform(upper_start, upper, (sample_size[0]//2, sample_size[1]))
        x = np.concatenate((x1, x2))
    else:
        x = np.random.uniform(lower, upper, sample_size)

    noise = np.random.normal(mu, sigma, size=(sample_size[0],))
    y = func(x, noise)
    return x, y

def generate_dataset(sample_shape, func, mu, sigma, sample_space=(-10, 10), void_space=None):
    x, y = data_gen(sample_shape, func, mu, sigma, sample_space, void_space)
    return x, y

'''
    Loads the data from the dataset_name folder and 
    returns the train, validation and test sets as torch tensors

    The data is expected to be in the following format:
    ./data
        - dataset_name
            - train.csv
            - val.csv
            - test.csv
    
    returns: (x_train, y_train), (x_val, y_val),  (x_test, y_test)
'''
def load_data(path, load_train=True, load_val=True, load_test=True):
    if load_train:
        train = genfromtxt(f"{path}/train.csv", delimiter=',')
        x_train, y_train = train[:,:-1], train[:,-1]
        x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)

    if load_val:
        val = genfromtxt(f"{path}/val.csv", delimiter=',')
        x_val, y_val = val[:,:-1], val[:,-1]
        x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)

    if load_test:
        test = genfromtxt(f"{path}/test.csv", delimiter=',')
        x_test, y_test = test[:,:-1], test[:,-1]
        x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)

        test_in_domain = genfromtxt(f"{path}/test_in_domain.csv", delimiter=',')
        x_test_in_domain, y_test_in_domain = test_in_domain[:,:-1], test_in_domain[:,-1]
        x_test_in_domain, y_test_in_domain = torch.Tensor(x_test_in_domain), torch.Tensor(y_test_in_domain)

        test_out_domain = genfromtxt(f"{path}/test_out_domain.csv", delimiter=',')
        x_test_out_domain, y_test_out_domain = test_out_domain[:,:-1], test_out_domain[:,-1]
        x_test_out_domain, y_test_out_domain = torch.Tensor(x_test_out_domain), torch.Tensor(y_test_out_domain)


    ret_train = (x_train, y_train) if load_train else None
    ret_val = (x_val, y_val) if load_val else None
    ret_test = (x_test, y_test) if load_test else None
    ret_test_in_domain = (x_test_in_domain, y_test_in_domain) if load_test else None
    ret_test_out_domain = (x_test_out_domain, y_test_out_domain) if load_test else None

    return ret_train, ret_val, ret_test, ret_test_in_domain, ret_test_out_domain