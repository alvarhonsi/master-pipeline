import torch
import torch.nn as nn
from collections import OrderedDict

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
            mods["drop0"] = nn.Dropout(0.5)
            for i in range(len(hidden_features)-1):
                mods["fc"+str(i+1)] = nn.Linear(hidden_features[i], hidden_features[i+1])
                mods["act"+str(i+1)] = nn.ReLU()
                mods["drop"+str(i+1)] = nn.Dropout(0.5)
            mods["fc"+str(len(hidden_features))] = nn.Linear(hidden_features[-1], out_features)

        self.fc = nn.Sequential(mods)

    def forward(self, x):
        out = self.fc(x)
        mu = out.squeeze()

        return mu
    

class BaselineNN(nn.Module):
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
            mods["drop0"] = nn.Dropout(0.5)
            for i in range(len(hidden_features)-1):
                mods["fc"+str(i+1)] = nn.Linear(hidden_features[i], hidden_features[i+1])
                mods["act"+str(i+1)] = nn.ReLU()
                mods["drop"+str(i+1)] = nn.Dropout(0.5)
            mods["fc"+str(len(hidden_features))] = nn.Linear(hidden_features[-1], out_features)

        self.fc = nn.Sequential(mods)

    def forward(self, x):
        out = self.fc(x)
        mu = out.squeeze()

        return mu
    