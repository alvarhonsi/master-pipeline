import numpy as np
from numpy import genfromtxt
import torch
import json
import scipy

def identity(x) -> np.array:
    return x

def linear_func(x) -> np.array:
    return 2*x + 1

def polynomial_func(x) -> np.array:
    return 0.5*x**2 - 3*x + 1

def sinusoidal_func(x) -> np.array:
    return np.sin(x) * 2 + 3


def normal_percentile(mu, sigma, percentile) -> float:
    return mu + sigma * scipy.stats.norm.ppf(percentile)

'''
    Dictionary of functions that can be used in data_gen
    functions should take a numpy array [n, x_dim] and return a numpy array [n] - the output of the function
'''
data_functions = {
    "identity": identity,
    "linear": linear_func,
    "polynomial": polynomial_func,
    "sinusoidal": sinusoidal_func,
    "sum": lambda xs : xs.sum(axis=1),
    "product": lambda xs : xs.prod(axis=1),
    "mean": lambda xs : xs.mean(axis=1),
    "max": lambda xs : xs.max(axis=1),
    "min": lambda xs : xs.min(axis=1),
    "sqrt_of_sum": lambda xs : np.sqrt(xs.sum(axis=1)),
    "euclidean_norm": lambda xs : np.sqrt((xs**2).sum(axis=1)), #sqrt of sum of squares
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
def data_gen(sample_size, func, mu, sigma, sample_space):
    try:
        func = data_functions[func]
    except KeyError:
        raise Exception("Invalid function name")

    lower, upper = sample_space

    x = np.random.uniform(lower, upper, sample_size)
    y = func(x) + np.random.normal(mu, sigma**2, sample_size[0])

    return x, y

def generate_dataset(sample_shape, func, mu, sigma, sample_space=(-10, 10)):
    x, y = data_gen(sample_shape, func, mu, sigma, sample_space)
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
def load_data(path):
    train = genfromtxt(f"{path}/train.csv", delimiter=',')
    val = genfromtxt(f"{path}/val.csv", delimiter=',')
    test = genfromtxt(f"{path}/test.csv", delimiter=',')

    x_train, y_train = train[:,:-1], train[:,-1]
    x_val, y_val = val[:,:-1], val[:,-1]
    x_test, y_test = test[:,:-1], test[:,-1]

    x_train, y_train = torch.Tensor(x_train), torch.Tensor(y_train)
    x_val, y_val = torch.Tensor(x_val), torch.Tensor(y_val)
    x_test, y_test = torch.Tensor(x_test), torch.Tensor(y_test)

    return (x_train, y_train), (x_val, y_val),  (x_test, y_test)