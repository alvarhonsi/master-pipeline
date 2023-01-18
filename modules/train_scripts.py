import torch
import pyro
import time
from datetime import timedelta
from pyro.infer import SVI, MCMC, NUTS, Trace_ELBO, Predictive

def eval_svi(svi, val_loader):
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            #calc loss and take gradient step
            loss = svi.evaluate_loss(x, y.flatten())
            val_loss += loss

    return val_loss / len(val_loader)


def eval_mcmc(mcmcbnn, val_loader):
    # Returns mean squared error
    # Mini-batch not supported for MCMC must use full dataset

    rmse = 0
    for x, y in val_loader:
        x = x
        y_true = y.flatten()
        y_pred = mcmcbnn.predict(x, num_predictions=100) # mean over samples is the predicted value
        rmse += torch.mean((y_pred - y_true) ** 2).sum()

    return (rmse / len(val_loader)).item() # mean rmse over batches