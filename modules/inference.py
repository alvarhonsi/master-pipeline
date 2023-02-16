import pyro
import torch
from pyro.infer import MCMC, SVI, Predictive, Trace_ELBO
import os
from collections import defaultdict
import time
from datetime import timedelta
from abc import ABC, abstractmethod

def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,

class BayesianInferenceModel(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def fit(self, dataloader):
        """Fits training data to inference model"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X, num_predictions=1):
        """Returns predictions for X"""
        raise NotImplementedError

    @abstractmethod
    def get_error(self, X, y, num_predictions=1):
        """Returns error for X and y (RMSE)"""
        raise NotImplementedError

    @abstractmethod
    def save(self, path):
        """Saves model to path"""
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        """Loads model from path"""
        raise NotImplementedError

class MCMCInferenceModel(BayesianInferenceModel):
    def __init__(self, model, kernel, num_samples=1000, num_chains=1, num_warmup=1000, device="cpu"):
        super().__init__(model)
        self.kernel = kernel
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.device = device

        self.mcmc = None
        self.samples = None

    def fit(self, dataloader):
        train_stats =  {
            "rmse": [],
            "val_rmse": [],
            "time": 0,
        }

        pyro.clear_param_store()
        start = time.time()

        #X, y = dataloader.dataset.tensors[0].to(self.device), dataloader.dataset.tensors[1].flatten().to(self.device)
        input_data_list = []
        observation_data_list = []
        for in_data, obs_data in dataloader:
            input_data_list.append(in_data.to(self.device))
            observation_data_list.append(obs_data.to(self.device))
        X = torch.cat(input_data_list)
        y = torch.cat(observation_data_list)

        
        self.mcmc = MCMC(self.kernel, num_samples=self.num_samples, num_chains=self.num_chains, warmup_steps=self.num_warmup)
        self.mcmc.run(X, y)
        self.samples = self.mcmc.get_samples()

        train_rmse = self.get_error(X, y)

        td = timedelta(seconds=round(time.time() - start))
        print(f"[{td}][mcmc finished] rmse: {round(train_rmse, 2)}")

        train_stats["rmse"].append(train_rmse)

        end = time.time()
        train_stats["time"] = end - start

        return train_stats

    def predict(self, X, num_predictions=1):
        if self.mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        X = X.to(self.device)

        weight_samples = self.mcmc.get_samples(num_samples=num_predictions)

        predictive = Predictive(self.model, weight_samples, return_sites=("obs", ))
        predictions = predictive(X)

        #Rotate prediction matrix to [x_samples, num_dist_samples]
        #y_pred = torch.transpose(predictions["obs"], 0, 1)
        y_pred = predictions["obs"]

        return y_pred

    def get_error(self, X, y, num_predictions=1):
        X, y = X.to(self.device), y.to(self.device)

        predictions = self.predict(X, num_predictions=num_predictions)
        rmse = torch.sqrt(torch.mean((predictions - y)**2))

        return rmse.item()

    def evaluate(self, dataloader, num_predictions=1):
        rmse = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)

            rmse += self.get_error(X, y, num_predictions=num_predictions)

        rmse /= len(dataloader)

        return rmse

    def save(self, path):
        if self.mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        self.mcmc.kernel.potential_fn = None
        torch.save({ "model": self.model.state_dict(), "mcmc": self.mcmc}, f"{path}/checkpoint.pt")
        print(f"Saved model and samples to {path}")

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")

        checkpoint = torch.load(f"{path}/checkpoint.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.mcmc = checkpoint["mcmc"]

        print(f"Loaded model and samples from {path}")

class SVIInferenceModel(BayesianInferenceModel):
    def __init__(self, model, guide, optim, epochs=100, num_steps=1000, loss=None, num_particles=1, device="cpu"):
        super().__init__(model)
        self.guide = guide
        self.optim = optim
        self.loss = loss if loss else Trace_ELBO(num_particles=num_particles)
        self.epochs = epochs
        self.num_steps = num_steps
        self.num_particles = num_particles
        self.device = device

        self.svi = None

    def fit(self, dataloader, callback=None, closed_form_kl=True):
        train_stats =  {
            "elbo": [],
            "rmse": [],
            "val_rmse": [],
            "time": 0,
        }

        pyro.clear_param_store()
        start = time.time()

        # Scale the loss to account for dataset size.
        self.model = pyro.poutine.scale(self.model, scale=1.0/len(dataloader.dataset))
        
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.loss)

        for epoch in range(1, self.epochs + 1):
            elbo = 0
            rmse = 0
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                elbo += self.svi.step(X, y)
                rmse += self.get_error(X, y, num_predictions=1)

            elbo = elbo / len(dataloader)
            rmse = rmse / len(dataloader)

            if epoch % 10 == 0 or epoch == 1:
                td = timedelta(seconds=round(time.time() - start))
                print(f"[{td}][epoch {epoch}] elbo: {elbo:.2f} rmse: {round(rmse, 2)}")

            if callback is not None and callback(elbo, epoch):
                break

            train_stats["elbo"].append(elbo)

        end = time.time()
        train_stats["time"] = end - start

        return train_stats

    def predict(self, X, num_predictions=1):
        if self.svi is None:
            raise RuntimeError("Call .fit to run SVI and obtain samples from the posterior first.")

        X = X.to(self.device)

        predictive = Predictive(self.model, guide=self.guide, num_samples=num_predictions, return_sites=("obs", ), parallel=True)
        predictions = predictive(X)

        #Rotate prediction matrix to [x_samples, num_dist_samples]
        #y_pred = torch.transpose(predictions["obs"], 0, 1)

        return predictions["obs"]

    def get_error(self, X, y, num_predictions=1):
        X, y = X.to(self.device), y.to(self.device)

        predictions = self.predict(X, num_predictions=num_predictions)
        rmse = torch.sqrt(torch.mean((predictions - y)**2))

        return rmse.item()

    def evaluate(self, dataloader, num_predictions=1):
        rmse = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            rmse += self.get_error(X, y, num_predictions=num_predictions)

        rmse /= len(dataloader)

        return rmse

    def save(self, path):
        if self.svi is None:
            raise RuntimeError("Call .fit to run SVI and obtain samples from the posterior first.")

        torch.save({"model": self.model.state_dict(), "guide": self.guide}, f"{path}/checkpoint.pt")
        pyro.get_param_store().save(f"{path}/params.pt")
        print(f"Saved model and parameters to {path}")

    def load(self, path):
        checkpoint = torch.load(f"{path}/checkpoint.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.guide = checkpoint["guide"]
        pyro.get_param_store().load(f"{path}/params.pt", map_location=self.device)

        self.svi = SVI(self.model, self.guide, self.optim, loss=self.loss)
        print(f"Loaded model and parameters from {path}")
        