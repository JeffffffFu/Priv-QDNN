#!/usr/bin/env python3
"""
LOOD-style memorization score (Ye et al., leave-one-out dataset formulation).

For a sample z in the training pool D: train one model on D and one on D\\{z},
query both at Q=z, and take
MemScore(z) = KL( softmax(f_D(z)) || softmax(f_{D^{-z}}(z)) ) (natural log, plus a
stabilizing term).

By default num_loo samples z are drawn at random; the mean and standard deviation of
their scores are reported and written to result/memorization_lood/.

Cost: one full training run plus num_loo leave-one-out runs. Enabling PTQ (hawq,
adaround, ...) makes a single LOO run much slower, so start with --quick or
--train_pool_size 2000 --num_loo 20 --num_epochs 5.

Run from the Priv-QDNN root:

  python experiment/memorization.py \\
    --dataset_name cifar100 --net_name resnet18 \\
    --quantization Original --device cuda:0 \\
    --train_pool_size 5000 --num_loo 100 --num_epochs 20 \\
    --seed 42

With PTQ (same as training/train.py; calibration comes from the current training pool)::

  python experiment/memorization.py \\
    --dataset_name cifar100 --net_name resnet18 \\
    --quantization PYT_PTQ --a_bit 8 --w_bit 8 --device cuda:0 \\
    --train_pool_size 4000 --num_loo 30 --num_epochs 15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.load_data import get_data  # noqa: E402
from model.DNN import DNN  # noqa: E402
from training.utlis import sample_target_samples  # noqa: E402


def _build_args_dict(ns: argparse.Namespace) -> dict:
    return {
        "device": ns.device,
        "dataset_name": ns.dataset_name,
        "net_name": ns.net_name,
        "quantization": (
            "Original" if ns.quantization == "Original" else ns.quantization
        ),
        "num_epochs": int(ns.num_epochs),
        "batch_size": int(ns.batch_size),
        "lr": float(ns.lr),
        "optim": ns.optim,
        "a_bit": int(ns.a_bit),
        "w_bit": int(ns.w_bit),
        "random": int(ns.seed),
        "attack_method": "MIA",
        "trials": 1,
        "eps": 0.0,
        "observations": 1,
        "compleaknr_attacks": "confidence",
        "training": "None",
    }


def _unwrap_for_forward(model_or_dnn):
    if isinstance(model_or_dnn, DNN):
        return model_or_dnn.model
    return model_or_dnn


def forward_logits(model_or_dnn, x: torch.Tensor, device: str) -> torch.Tensor:
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device, dtype=torch.float32)
    net = _unwrap_for_forward(model_or_dnn)
    net.eval()
    with torch.no_grad():
        return net(x)


def softmax_kl_div(p_logits: torch.Tensor, q_logits: torch.Tensor, eps: float = 1e-8) -> float:
    """KL( softmax(p) || softmax(q) ) over the first row of the batch (a single sample)."""
    p = F.softmax(p_logits, dim=-1).clamp_min(eps)
    q = F.softmax(q_logits, dim=-1).clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean().item()


def train_dnn_on_subset(subset: Subset, test_data, args_dict: dict) -> DNN:
    dnn = DNN(args_dict)
    train_loader = DataLoader(
        subset, batch_size=args_dict["batch_size"], shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=args_dict["batch_size"], shuffle=False, num_workers=0
    )
    dnn.train_model(train_loader, test_loader, label_key=None)
    return dnn


def maybe_apply_ptq(
    dnn: DNN, calib_base: Subset, args_dict: dict
):
    if args_dict["quantization"] == "Original":
        return dnn
    from quantization import get_PTQ_quantization

    PTQ = get_PTQ_quantization(args_dict["quantization"], args_dict["net_name"])
    q = args_dict["quantization"]
    if q in ("pact", "PYT_QAT", "hawq", "bsq"):
        sub = calib_base
    else:
        sub = sample_target_samples(
            calib_base, min(4096, len(calib_base)), str(args_dict["random"])
        )
    return PTQ(dnn, sub, args_dict)


def main():
    parser = argparse.ArgumentParser(description="LOOD memorization score (KL on self-query)")
    parser.add_argument("--dataset_name", type=str, default="cifar100")
    parser.add_argument("--net_name", type=str, default="resnet18")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_loo", type=int, default=100, help="number of leave-one-out targets z drawn at random")
    parser.add_argument(
        "--train_pool_size",
        type=int,
        default=5000,
        help="training pool |D|: take this many samples after shuffling the training set; 0 uses the full training set (very slow)",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optim", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--quantization", type=str, default="Original")
    parser.add_argument("--a_bit", type=int, default=8)
    parser.add_argument("--w_bit", type=int, default=8)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke run: train_pool_size=800, num_loo=5, num_epochs=3",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default=None,
        help="defaults to result/memorization_lood/<dataset>/<net>/...",
    )
    ns = parser.parse_args()

    if ns.quick:
        ns.train_pool_size = 800
        ns.num_loo = 5
        ns.num_epochs = 3
        print("[memorization_lood] --quick: small-scale smoke-run settings enabled.")

    args_dict = _build_args_dict(ns)

    if ns.out_root is None:
        out_dir = os.path.join(
            _ROOT,
            "result",
            "memorization_lood",
            ns.dataset_name,
            ns.net_name,
            f"{ns.quantization}_{ns.a_bit}_{ns.w_bit}",
            datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
    else:
        out_dir = ns.out_root
    os.makedirs(out_dir, exist_ok=True)

    train_data, test_data = get_data(ns.dataset_name)

    rng = np.random.RandomState(ns.seed)
    n_train = len(train_data)
    perm = rng.permutation(n_train).tolist()
    if ns.train_pool_size and ns.train_pool_size > 0:
        pool = perm[: min(ns.train_pool_size, n_train)]
    else:
        pool = perm

    if len(pool) < ns.num_loo + 2:
        raise ValueError(
            f"Training pool too small: |pool|={len(pool)}; it must exceed num_loo+1 so samples remain after leaving one out."
        )

    num_loo_eff = min(ns.num_loo, len(pool) - 1)
    loo_positions = rng.choice(len(pool), size=num_loo_eff, replace=False).tolist()

    D_full = Subset(train_data, pool)

    print(f"training pool |D|={len(pool)}, leave-one-out samples={num_loo_eff}, epochs={ns.num_epochs}")
    print(f"output directory: {out_dir}")

    # ---- f_D ----
    print(">>> training f_D (full pool D) ...")
    dnn_full = train_dnn_on_subset(D_full, test_data, args_dict)
    calib_subset = D_full
    model_full = maybe_apply_ptq(dnn_full, calib_subset, args_dict)
    del dnn_full

    scores: list[float] = []
    meta_rows: list[dict] = []

    for k, pos in enumerate(tqdm(loo_positions, desc="LOOD leave-one-out")):
        z_idx_global = pool[pos]
        other = [pool[j] for j in range(len(pool)) if j != pos]
        D_minus = Subset(train_data, other)

        print(f"  [{k+1}/{num_loo_eff}] training f_{{D\\z}}, |D|={len(other)} ...")
        dnn_loo = train_dnn_on_subset(D_minus, test_data, args_dict)
        calib_loo = D_minus
        model_loo = maybe_apply_ptq(dnn_loo, calib_loo, args_dict)
        del dnn_loo

        x_z, y_z = train_data[z_idx_global]
        if not torch.is_tensor(x_z):
            x_z = torch.as_tensor(x_z)
        lf = forward_logits(model_full, x_z.clone(), ns.device)
        ll = forward_logits(model_loo, x_z.clone(), ns.device)
        score = softmax_kl_div(lf, ll)
        scores.append(score)
        meta_rows.append(
            {
                "k": k,
                "pool_position": int(pos),
                "global_index": int(z_idx_global),
                "label": int(y_z) if not isinstance(y_z, dict) else None,
                "mem_kl": float(score),
            }
        )
        del model_loo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores_np = np.array(scores, dtype=np.float64)
    summary = {
        "dataset_name": ns.dataset_name,
        "net_name": ns.net_name,
        "quantization": ns.quantization,
        "a_bit": ns.a_bit,
        "w_bit": ns.w_bit,
        "train_pool_size_effective": len(pool),
        "num_loo": num_loo_eff,
        "num_epochs": ns.num_epochs,
        "seed": ns.seed,
        "mean_mem_kl": float(scores_np.mean()) if len(scores_np) else float("nan"),
        "std_mem_kl": float(scores_np.std(ddof=0)) if len(scores_np) else float("nan"),
        "dist_metric": "KL( P_D || P_D_minus_z ) on softmax outputs at Q=z",
        "note": "Higher means z is more sensitive to being in the training set, i.e. more strongly memorized (consistent with the LOOD account in Ye et al.).",
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(out_dir, "mem_scores.npy"), scores_np)

    import csv

    per_path = os.path.join(out_dir, "per_sample.csv")
    if meta_rows:
        with open(per_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)
    else:
        open(per_path, "w", encoding="utf-8").close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
