#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.WhiteBox_MIA import _get_eval_data  # noqa: E402
from data.prepare_data import split_dataset  # noqa: E402
from extract_last_layer_grad_norm import (  # noqa: E402
    _default_model_path,
    _find_last_classifier_weight_name,
    _unwrap_torch_model,
    compute_grad_l2_last_layer,
)
from training.utlis import sample_target_samples  # noqa: E402

ROWS = [
    ("Full-precision", "Original", 32, 32),
    ("Pytorch-Quant", "PYT_PTQ", None, None),
    ("AdaRound", "adaround", None, None),
    ("BSQ", "bsq", None, None),
    ("PACT", "pact", None, None),
    ("QDrop", "qdrop", None, None),
    ("HAWQ", "hawq", None, None),
]
BITS = (4, 6, 8, 10)


def _result_base(dataset_name: str, net_name: str) -> Path:
    return Path(_ROOT) / "result" / "grad_norm_range_norm_table" / dataset_name / net_name


def _cfg(attack_method: str, net_name: str, dataset_name: str, quantization: str, a_bit: int, w_bit: int, random: str, eps: float) -> dict:
    return {
        "attack_method": attack_method,
        "net_name": net_name,
        "dataset_name": dataset_name,
        "quantization": quantization,
        "a_bit": int(a_bit),
        "w_bit": int(w_bit),
        "eps": float(eps),
        "random": random,
    }


def _load_net(cfg: dict, trial: int, device: str):
    path = _default_model_path(cfg, trial)
    print(f"  loading: {path}")
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    return _unwrap_torch_model(loaded), path


def _mem_nonmem(dataset_name: str, random: str):
    train_data, test_data = _get_eval_data(dataset_name)
    rnd = None if random == "None" else int(random)
    target_m, _, _ = split_dataset(train_data, rnd)
    t_n = int(0.2 * len(target_m))
    return sample_target_samples(target_m, t_n, random), sample_target_samples(test_data, t_n, random)


def _range_normalized_diff(mem: np.ndarray, non: np.ndarray, eps: float = 1e-12) -> float:
    allv = np.concatenate([mem.ravel(), non.ravel()])
    den = float(np.max(allv) - np.min(allv))
    if den < eps:
        return 0.0
    return float((np.mean(non) - np.mean(mem)) / den)


def _range_normalized_details(mem: np.ndarray, non: np.ndarray, eps: float = 1e-12) -> dict:
    allv = np.concatenate([mem.ravel(), non.ravel()])
    lo = float(np.min(allv))
    hi = float(np.max(allv))
    den = hi - lo
    mean_mem = float(np.mean(mem))
    mean_non = float(np.mean(non))
    num = mean_non - mean_mem
    score = 0.0 if den < eps else float(num / den)
    return {
        "mean_mem": mean_mem,
        "mean_non": mean_non,
        "numerator_non_minus_mem": num,
        "min_mem_or_non": lo,
        "max_mem_or_non": hi,
        "denominator_range": den,
        "score": score,
    }


def _one_cell(cfg: dict, trial: int, device: str, mem, nm, eps: float):
    net, path = _load_net(cfg, trial, device)
    pname = _find_last_classifier_weight_name(net)
    gm = compute_grad_l2_last_layer(net, mem, device, param_name=pname)
    gnm = compute_grad_l2_last_layer(net, nm, device, param_name=pname)
    details = _range_normalized_details(gm, gnm, eps=eps)
    return details["score"], {"path": path, "param": pname}, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack_method", default="MIA")
    ap.add_argument("--net_name", default="resnet18")
    ap.add_argument("--dataset_name", default="cifar100")
    ap.add_argument("--random", default="42")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--full_table", action="store_true")
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--a_bit", type=int, default=None)
    ap.add_argument("--w_bit", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, using CPU")

    mem, nm = _mem_nonmem(args.dataset_name, args.random)
    base = _result_base(args.dataset_name, args.net_name)
    if args.full_table:
        out = Path(args.out_dir) if args.out_dir else base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        if not args.quantization or args.a_bit is None or args.w_bit is None:
            raise ValueError("single-cell mode needs --quantization --a_bit --w_bit")
        out = Path(args.out_dir) if args.out_dir else base / f"{args.quantization}_{args.a_bit}_{args.w_bit}"
    out.mkdir(parents=True, exist_ok=True)

    metric = "(mean(non)-mean(mem)) / (max(mem∪non)-min(mem∪non))"
    if not args.full_table:
        cfg = _cfg(args.attack_method, args.net_name, args.dataset_name, args.quantization, args.a_bit, args.w_bit, args.random, 0.0)
        d, meta, details = _one_cell(cfg, args.trial, device, mem, nm, args.eps)
        row = {
            "Type": args.quantization,
            f"{args.a_bit}/{args.w_bit}": d,
            "metric": metric,
            "metric_values": details,
            "out_dir": str(out),
            "meta": meta,
        }
        print(json.dumps(row, indent=2, ensure_ascii=False))
        with open(out / "single_cell.json", "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
        return

    table: dict[str, list] = {"Type": []}
    for b in BITS:
        table[f"{b}/{b}"] = []
    table["32/32 (FP)"] = []
    table["Type"].append("Full-precision")
    for b in BITS:
        table[f"{b}/{b}"].append(float("nan"))
    cfg_fp = _cfg(args.attack_method, args.net_name, args.dataset_name, "Original", 32, 32, args.random, 0.0)
    try:
        d_fp, _, _ = _one_cell(cfg_fp, args.trial, device, mem, nm, args.eps)
    except Exception:
        d_fp = float("nan")
    table["32/32 (FP)"].append(d_fp)
    for lab, qname, _, _ in ROWS[1:]:
        table["Type"].append(lab)
        for b in BITS:
            cfg = _cfg(args.attack_method, args.net_name, args.dataset_name, qname, b, b, args.random, 0.0)
            try:
                d, _, _ = _one_cell(cfg, args.trial, device, mem, nm, args.eps)
            except Exception:
                d = float("nan")
            table[f"{b}/{b}"].append(d)
        table["32/32 (FP)"].append(float("nan"))
    df = pd.DataFrame(table)
    df.to_csv(out / "grad_norm_diff_table.csv", index=False)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"metric": metric, "dataset_name": args.dataset_name, "net_name": args.net_name}, f, indent=2, ensure_ascii=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
