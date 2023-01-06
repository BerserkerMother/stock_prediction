import math

import torch
from torch.utils import data

import numpy as np
import pandas as pd


class StockDataset(data.Dataset):
    def __init__(self, root: str):
        self.root = root
        df = pd.read_csv(root)
        close = df["Close"].to_numpy()
        features = df.iloc[:, 5:]
        features = features.to_numpy()
        self.features = np.concatenate([close.reshape(-1, 1) / 20000, features / 100], axis=1)
        self.target = [prepare_labels(close[i - 1], close[i]) for i in range(len(close))]

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float)
        target = self.target[idx]
        return features, target[0], target[1]


def prepare_labels(pri, pos):
    percent = math.fabs(pri - pos) / pri
    if pri > pos:
        return 0., percent
    return 1., percent
