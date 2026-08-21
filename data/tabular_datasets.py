"""Tabular datasets used in membership-inference literature (Texas100, Location).

Expected layout after preparation:

  data/texas/train_feats.npy, train_labels.npy, test_feats.npy, test_labels.npy
  data/location/train_feats.npy, train_labels.npy, test_feats.npy, test_labels.npy

Raw archives (download manually):
  - Texas100: https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
  - Location: https://github.com/privacytrustlab/datasets (location set)

Run `python data/prepare_tabular.py --dataset texas` after placing raw files.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import TensorDataset


TABULAR_META = {
    "texas": {"input_dim": 6169, "num_classes": 100},
    "location": {"input_dim": 168, "num_classes": 8},
}


def _load_npy_pair(feats_path: str, labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(feats_path) or not os.path.isfile(labels_path):
        raise FileNotFoundError(
            f"Missing tabular files:\n  {feats_path}\n  {labels_path}\n"
            f"See data/tabular_datasets.py header for download/prepare instructions."
        )
    x = np.load(feats_path)
    y = np.load(labels_path)
    return x.astype(np.float32), y.astype(np.int64)


def _to_tensor_dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y))


def get_tabular_data(name: str) -> Tuple[TensorDataset, TensorDataset]:
    if name not in TABULAR_META:
        raise ValueError(f"Unknown tabular dataset {name}")

    root = os.path.join(os.getcwd(), "data", name)
    train_x, train_y = _load_npy_pair(
        os.path.join(root, "train_feats.npy"),
        os.path.join(root, "train_labels.npy"),
    )
    test_x, test_y = _load_npy_pair(
        os.path.join(root, "test_feats.npy"),
        os.path.join(root, "test_labels.npy"),
    )

    meta = TABULAR_META[name]
    if train_x.shape[1] != meta["input_dim"]:
        raise ValueError(
            f"{name} train feature dim {train_x.shape[1]} != expected {meta['input_dim']}"
        )

    num_classes = int(max(train_y.max(), test_y.max()) + 1)
    if num_classes != meta["num_classes"]:
        print(
            f"[tabular] Warning: {name} label count={num_classes}, "
            f"expected {meta['num_classes']}; using inferred count."
        )
        TABULAR_META[name]["num_classes"] = num_classes

    return _to_tensor_dataset(train_x, train_y), _to_tensor_dataset(test_x, test_y)
