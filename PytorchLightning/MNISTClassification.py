# import libraries

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd

import os
from git import Repo

if not os.path.exists("mnist-pngs"):
    Repo.clone_from("https://github.com/rasbt/mnist-pngs", "mnist-pngs")

url = "https://github.com/rasbt/mnist-pngs"
df_train = pd.read_csv("mnist-pngs/train.csv")
df_test = pd.read_csv("mnist-pngs/test.csv")


# split data into validation and train
df_val = df_train.sample(frac=0.2, random_state=42)
df_train = df_train.drop(df_val.index)

df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)


# create dataset loader
class MNISTDataset(Dataset):
    def __init__(self, csv_path) -> None:
        super().__init__()


# create a MLP classifer
class MLPClassifier(nn.Module):
    def __init__(self, num_features) -> None:
        super().__init__()
        self.num_features = num_features

        # write model architecture
        self.model_arch = nn.Sequential(
            nn.Linear(self.num_features, 50, dtype=torch.float64),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        logits = self.model_arch(x)
        return logits
