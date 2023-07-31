import torch
import torch.nn as nn
from collections import OrderedDict


def _as_tuple(x):
    if isinstance(x, (list, tuple)):
        return x
    return x,


class FFNN(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], device="cpu"):
        super().__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        mods = OrderedDict()

        if hidden_features == []:
            mods["fc0"] = nn.Linear(in_features, out_features)
        else:
            mods["fc0"] = nn.Linear(in_features, hidden_features[0])
            mods["act0"] = nn.ReLU()
            #mods["drop0"] = nn.Dropout(0.5)
            for i in range(len(hidden_features)-1):
                mods["fc"+str(i+1)] = nn.Linear(hidden_features[i],
                                                hidden_features[i+1])
                mods["act"+str(i+1)] = nn.ReLU()
                #mods["drop"+str(i+1)] = nn.Dropout(0.5)
            mods["fc"+str(len(hidden_features))
                 ] = nn.Linear(hidden_features[-1], out_features)

        self.fc = nn.Sequential(mods)

    def forward(self, x):
        out = self.fc(x)
        mu = out.squeeze()

        return mu


class BaselineNN(nn.Module):
    def __init__(self, net, device="cpu"):
        super().__init__()
        self.net = net.to(device)
        self.device = device

    def fit(self, data_loader, optim, num_epochs, callback=None, device=None):
        old_training_state = self.net.training
        self.net.train(True)

        loss_fn = nn.MSELoss()

        for i in range(num_epochs):
            loss = 0.
            num_batch = 1
            for num_batch, (input_data, observation_data) in enumerate(iter(data_loader), 1):
                loss += self.step(input_data.to(device),
                                  observation_data.to(device), optim, loss_fn)

            if callback is not None and callback(self, i, loss / num_batch):
                break

        self.net.train(old_training_state)
        return self.net

    def step(self, input_data, observation_data, optim, loss_fn):
        optim.zero_grad()

        out = self.net(input_data)

        loss = loss_fn(out, observation_data)
        loss.backward()

        optim.step()

        return loss.item()

    def predict(self, *input_data, num_predictions=1, aggregate=False):
        old_training_state = self.net.training
        self.net.train(False)
        with torch.autograd.no_grad():
            preds = self.net(*input_data)
            scales = torch.full_like(preds, 0., device=self.device)

        self.net.train(old_training_state)
        return preds, scales if aggregate else preds

    def evaluate(self, input_data, obs_data, num_predictions=1, reduction="sum"):
        """"Utility method for evaluation. Calculates a likelihood-dependent errors measure, e.g. squared errors or
        mis-classifications and

        :param input_data: Inputs to the neural net. Must be a tuple of more than one.
        :param y: observations, e.g. class labels.
        :param int num_predictions: number of forward passes.
        :param bool aggregate: whether to aggregate the outputs of the forward passes before evaluating.
        :param str reduction: "sum", "mean" or "none". How to process the tensor of errors. "sum" adds them up,
            "mean" averages them and "none" simply returns the tensor."""

        preds = self.predict(*_as_tuple(input_data), aggregate=True)[0]
        mse = torch.mean((preds - obs_data)**2, dim=0)

        return mse

    def get_likelihood_scale(self, *input_data, num_predictions=1):
        x = input_data[0][0]

        scales = torch.zeros(
            num_predictions, x.shape[0], device=self.device)

        mean = scales.mean(dim=0)
        std = scales.std(dim=0)
        return mean, std

    def sample_predictive(self, *input_data, num_predictions=1, guide_traces=None):
        if guide_traces is None:
            guide_traces = [None] * num_predictions

        preds = []
        with torch.autograd.no_grad():
            for trace in guide_traces:
                pred = self.predict(*input_data, aggregate=True)[0]
                preds.append(pred)
        predictions = torch.stack(preds)
        return predictions

    def get_error_metrics(self, input_data, y, num_predictions=1, aggregate=True, reduction="mean"):

        preds = self.predict(*_as_tuple(input_data), aggregate=True)[0]
        mse = torch.mean((preds - y)**2, dim=0)
        mae = torch.mean(torch.abs(preds - y), dim=0)

        return mse, torch.tensor(0), mae
