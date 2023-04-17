import pyro

optimizer_types = {
    "adam": pyro.optim.Adam,
    "sgd": pyro.optim.SGD,
    "rmsprop": pyro.optim.RMSprop,
}