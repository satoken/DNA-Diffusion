import os
import pickle
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from dnadiffusion.utils.utils import one_hot_encode


def load_data(
    df: pd.DataFrame,
    max_seq_len: int = 200,
    right_aligned: bool = False,
    tag_name: list[str] = ["TAG"],
):
    # Creating sequence dataset
    nucleotides = ["A", "C", "G", "T"]
    valid_indices = df["sequence"].apply(lambda x: "N" not in x)
    x_train_seq = np.array([one_hot_encode(x, nucleotides, max_seq_len=max_seq_len, right_aligned=right_aligned) for x in df["sequence"][valid_indices]])
    X_train = np.array([x.T.tolist() for x in x_train_seq])
    X_train[X_train == 0] = -1

    # Creating labels
    tags = [ ]
    x_train_cell_type = []
    for tag in tag_name:
        tag_to_numeric = {x: n for n, x in enumerate(df[tag][valid_indices].unique(), 1)}
        numeric_to_tag = dict(enumerate(df[tag][valid_indices].unique(), 1))
        cell_types = list(numeric_to_tag.keys())
        x_train_cell_type.append(torch.tensor([tag_to_numeric[x] for x in df[tag][valid_indices]]))

        tags.append(
            {
                "name": tag,
                "tag_to_numeric": tag_to_numeric,
                "numeric_to_tag": numeric_to_tag,
                "cell_types": cell_types,
            }
        )

    # Collecting variables into a dict
    encode_data_dict = {
        "X_train": X_train,
        "x_train_cell_type": torch.stack(x_train_cell_type, dim=1),
        "tags": tags,
    }

    return encode_data_dict


class SequenceDataset(Dataset):
    def __init__(
        self,
        seqs: np.ndarray,
        c: torch.Tensor,
        transform: T.Compose = T.Compose([T.ToTensor()]),
    ):
        "Initialization"
        self.seqs = seqs
        self.c = c
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.seqs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        image = self.seqs[index]

        if self.transform:
            x = self.transform(image)
        else:
            x = image

        y = self.c[index]

        return x, y
