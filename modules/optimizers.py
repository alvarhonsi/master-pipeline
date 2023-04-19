import pyro
from pyro.optim.optim import ClippedAdam


optimizer_types = {
    "adam": pyro.optim.Adam,
    "sgd": pyro.optim.SGD,
    "rmsprop": pyro.optim.RMSprop,
    "clipped-adam": ClippedAdam,
}