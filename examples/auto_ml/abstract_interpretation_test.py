from typing import Any

import pandas as pd
import numpy as np
import torch
from torch import nn

from collections.abc import Container

from cl3s import DSL, Constructor, Literal, Type, Var, SearchSpaceSynthesizer

from typing import Any


def load_iris(path):
    iris = pd.read_csv(path)

    # load training data
    train_input = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy()

    # construct labels manually since data is ordered by class
    train_labels = np.array([0]*50 + [1]*50 + [2]*50).reshape(-1)

    # one-hot encode 3 classes
    train_labels = np.identity(3)[train_labels]

    return train_input, train_labels


class Abstract_Interpretation_Repository:

    def __init__(self, learning_rates: list[float],
                 batch_sizes: list[int],
                 epochs: list[int],
                 dataset: torch.utils.data.TensorDataset):
        self.learning_rates = learning_rates
        self.batch_size = batch_sizes
        self.epochs = epochs
        self.train_dataset = dataset

    def parameter(self) -> dict[str, list[Any]]:
        return {
            "bool": [True, False],
            "learning_rate": self.learning_rates,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    def gamma(self):
        return {
        }


