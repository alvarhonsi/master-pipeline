import pyro
import torch
from pyro.infer import MCMC, SVI, Predictive, Trace_ELBO
import os
from collections import defaultdict
import time
from datetime import timedelta
from abc import ABC, abstractmethod
from tqdm.auto import trange, tqdm

class BayesianInferenceModel(ABC):
    def __init__(self, model):
        self.model = model
    
    @abstractmethod
    def fit(self, dataloader):
        """Fits training data to inference model"""
        raise NotImplementedError

    def predict(self, X, num_predictions=1):
        X = X.to(self.device)
        predictive = self.get_predictive(num_predictions=num_predictions)
        predictions = predictive(X)

        return predictions["obs"]

    def get_rmse(self, X, y, num_predictions=100):
        X, y = X.to(self.device), y.to(self.device)

        predictions = self.predict(X, num_predictions=num_predictions)
        mean_predictions = torch.mean(predictions, dim=0)
        rmse = (mean_predictions - y).pow(2).mean().sqrt()

        return rmse.item()

    def get_mae(self, X, y, num_predictions=100):
        X, y = X.to(self.device), y.to(self.device)

        predictions = self.predict(X, num_predictions=num_predictions)
        mean_predictions = torch.mean(predictions, dim=0)
        mae = (mean_predictions - y).abs().mean()

        return mae.item()

    def evaluate(self, dataloader, metric="rmse", num_predictions=100):
        error = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            if metric == "rmse":
                error += self.get_rmse(X, y, num_predictions=num_predictions)
            elif metric == "mae":
                error += self.get_mae(X, y, num_predictions=num_predictions)
            else:
                raise ValueError(f"Invalid metric: {metric}")

        error /= len(dataloader)

        return error

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

    def fit(self, dataloader, time_steps=False):
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

        train_rmse = self.get_rmse(X, y)

        td = timedelta(seconds=round(time.time() - start))
        print(f"[{td}][mcmc finished] rmse: {round(train_rmse, 2)}")

        train_stats["rmse"].append(train_rmse)

        end = time.time()
        train_stats["time"] = end - start

        return train_stats

    def get_predictive(self, num_predictions=1):
        if self.mcmc is None:
            raise RuntimeError("Call .fit to run MCMC and obtain samples from the posterior first.")

        weight_samples = self.mcmc.get_samples(num_samples=num_predictions)
        predictive = Predictive(self.model, weight_samples, return_sites=("obs", "_RETURN"))
        return predictive

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

    def fit(self, dataloader, callback=None, closed_form_kl=True, time_steps=False):
        train_stats =  {
            "elbo_minibatch": [],
            "elbo_epoch": [],
            "rmse_minibatch": [],
            "rmse_epoch": [],
            "time": 0,
        }

        pyro.clear_param_store()
        start = time.time()

        # Scale the loss to account for dataset size.
        # Scale etter num parameters?

        #self.model = pyro.poutine.scale(self.model, scale=1.0/len(dataloader.dataset))
        
        self.svi = SVI(self.model, self.guide, self.optim, loss=self.loss)

        bar = trange(self.epochs, desc="Epoch", leave=True)
        for epoch in bar:
            elbo = 0
            rmse = 0
            for X, y in dataloader:
                ms = time.time()
                print("start minibatch", ms)
                X, y = X.to(self.device), y.to(self.device)
                print("data to device", time.time() - ms)
                loss = self.svi.step(X, y)
                print("step", time.time() - ms)
                error = self.get_rmse(X, y, num_predictions=1)
                print("rmse", time.time() - ms)
                elbo += loss
                rmse += error

                print("end minibatch", time.time() - ms)
                print("")

                train_stats["elbo_minibatch"].append(loss)
                train_stats["rmse_minibatch"].append(error)

            elbo = elbo / len(dataloader)
            rmse = rmse / len(dataloader)

            bar.set_description(f'Training: [EPOCH {epoch}]')
            bar.set_postfix(loss=f'{loss:.3f}', rmse=f'{rmse:.3f}')

            train_stats["elbo_epoch"].append(elbo)
            train_stats["rmse_epoch"].append(rmse)

            if callback is not None and callback(elbo, epoch):
                break

        end = time.time()
        train_stats["time"] = end - start

        return train_stats

    def get_predictive(self, num_predictions=1):
        if self.svi is None:
            raise RuntimeError("Call .fit to run SVI and obtain samples from the posterior first.")
            
        predictive = Predictive(self.model, guide=self.guide, num_samples=num_predictions, return_sites=("obs", "_RETURN", "sigma"))
        return predictive

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
        