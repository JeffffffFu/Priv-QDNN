#!/usr/bin/env python3
"""Prepare Texas100 / Location tabular datasets into train/test .npy files."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def _save_split(out_dir: str, x: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    np.save(os.path.join(out_dir, "train_feats.npy"), x_train.astype(np.float32))
    np.save(os.path.join(out_dir, "train_labels.npy"), y_train.astype(np.int64))
    np.save(os.path.join(out_dir, "test_feats.npy"), x_test.astype(np.float32))
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test.astype(np.int64))
    print(
        f"Saved {out_dir}: train={len(x_train)}, test={len(x_test)}, "
        f"feat_dim={x_train.shape[1]}, classes={int(y.max()) + 1}"
    )


def _load_privacytrustlab_texas(raw_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Texas100 from privacytrustlab/datasets archive layout."""
    feats_path = os.path.join(raw_dir, "texas", "100", "feats")
    labels_path = os.path.join(raw_dir, "texas", "100", "labels")
    if not (os.path.isfile(feats_path) and os.path.isfile(labels_path)):
        raise FileNotFoundError(
            f"Expected privacytrustlab Texas files:\n  {feats_path}\n  {labels_path}"
        )
    x = np.loadtxt(feats_path, delimiter=",", dtype=np.float32)
    y = np.loadtxt(labels_path, dtype=np.int64)
    if y.min() >= 1:
        y = y - 1
    return x, y


def _load_privacytrustlab_location(raw_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Location from privacytrustlab/datasets `bangkok` file."""
    bangkok_path = os.path.join(raw_dir, "bangkok")
    if not os.path.isfile(bangkok_path):
        raise FileNotFoundError(f"Expected privacytrustlab Location file: {bangkok_path}")

    rows = []
    with open(bangkok_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = [p.strip().strip('"') for p in line.strip().split(",")]
            rows.append([float(p) for p in parts])

    arr = np.asarray(rows, dtype=np.float32)
    y_raw = arr[:, 0].astype(np.int64)
    x = arr[:, 1:169]
    # Map 30 original categories to 8 classes (5010 samples total).
    y = ((y_raw - 1) * 8 // 30).astype(np.int64)
    return x, y


def _load_raw_texas(raw_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    candidates = [
        ("feats.npy", "labels.npy"),
        ("X.npy", "y.npy"),
        ("texas100_feats.npy", "texas100_labels.npy"),
    ]
    for fx, fy in candidates:
        fx_path, fy_path = os.path.join(raw_dir, fx), os.path.join(raw_dir, fy)
        if os.path.isfile(fx_path) and os.path.isfile(fy_path):
            return np.load(fx_path), np.load(fy_path)

    pkl_path = os.path.join(raw_dir, "texas100.pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj["X"], obj["y"]
        return obj[0], obj[1]

    if os.path.isfile(os.path.join(raw_dir, "texas", "100", "feats")):
        return _load_privacytrustlab_texas(raw_dir)

    raise FileNotFoundError(
        f"No recognized Texas raw files under {raw_dir}. "
        "Place feats.npy/labels.npy, texas100.pkl, or extract dataset_texas.tgz here."
    )


def _load_raw_location(raw_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    candidates = [
        ("feats.npy", "labels.npy"),
        ("X.npy", "y.npy"),
        ("location_feats.npy", "location_labels.npy"),
    ]
    for fx, fy in candidates:
        fx_path, fy_path = os.path.join(raw_dir, fx), os.path.join(raw_dir, fy)
        if os.path.isfile(fx_path) and os.path.isfile(fy_path):
            return np.load(fx_path), np.load(fy_path)

    pkl_path = os.path.join(raw_dir, "location.pkl")
    if os.path.isfile(pkl_path):
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj["X"], obj["y"]
        return obj[0], obj[1]

    if os.path.isfile(os.path.join(raw_dir, "bangkok")):
        return _load_privacytrustlab_location(raw_dir)

    raise FileNotFoundError(
        f"No recognized Location raw files under {raw_dir}. "
        "Place feats.npy/labels.npy, location.pkl, or extract dataset_location.tgz here."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["texas", "location", "both"], required=True)
    parser.add_argument("--raw_root", default=None, help="Defaults to data/<dataset>/raw")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    datasets = ["texas", "location"] if args.dataset == "both" else [args.dataset]
    for name in datasets:
        raw_dir = args.raw_root or os.path.join(os.getcwd(), "data", name, "raw")
        out_dir = os.path.join(os.getcwd(), "data", name)
        if name == "texas":
            x, y = _load_raw_texas(raw_dir)
        else:
            x, y = _load_raw_location(raw_dir)
        _save_split(out_dir, x, y, args.test_size, args.seed)


if __name__ == "__main__":
    main()
