#!/usr/bin/env python3
"""
Effect of quantization on the L2 norm of the last-layer classifier gradient:
reports (mean over non-members - mean over members), matching the style of Table 5
in the paper.

Optional: before taking the difference, jointly min-max scale the norms of the
**full-precision** model (Original_32_32) and of each quantized configuration into
[0,1] using the same lo/hi. This is one realization of "normalize ... into the same
range" as described in the text.

Reuses the per-sample gradient norms from ``extract_last_layer_grad_norm`` and the
same sampling as WhiteBox.

Run from the Priv-QDNN root::

  python experiment/grad_norm_quantization_table.py --full_table --device cuda:0

Output layout (kept separate so runs do not overwrite each other):

- shared root: ``result/grad_norm_quant_table/<dataset>/<net>/``
- **FP norm cache** (reused for the same random/trial, so it is not recomputed every
  time): ``.../<net>/_fp_cache/r<random>_t<trial>/cache_fp_*.npy``
- **single cell**: ``.../<net>/<quant>_<a>_<w>/single_cell.json`` (one subdirectory per
  quantization method / bit-width)
- **full table**: ``.../<net>/run_<YYYYMMDD_HHMMSS>/grad_norm_diff_table.csv`` (a new
  subdirectory per full-table run).
  With normalization on, the first row ``32/32 (FP)`` is the normalized difference
  under the FP norm span alone; every other cell is the ``(non-member - member)`` mean
  difference divided by the joint FP-union-Q span of **that cell**. The denominators
  generally differ from the one in the first row's last column, so those numbers should
  not be compared as absolute values.
"""

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

# Rows: display name -> (quantization name used under save/, a_bit, w_bit); apart from FP the bits come from the column
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
    return Path(_ROOT) / "result" / "grad_norm_quant_table" / dataset_name / net_name


def _fp_cache_dir(base: Path, random: str, trial: int) -> Path:
    rslug = str(random).replace(os.sep, "_").replace("/", "_")
    d = base / "_fp_cache" / f"r{rslug}_t{int(trial)}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cfg(
    attack_method: str,
    net_name: str,
    dataset_name: str,
    quantization: str,
    a_bit: int,
    w_bit: int,
    random: str,
    eps: float,
) -> dict:
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
    m = sample_target_samples(target_m, t_n, random)
    nm = sample_target_samples(test_data, t_n, random)
    return m, nm


def _joint_minmax(mem_fp, non_fp, mem_q, non_q, eps=1e-12):
    v = np.concatenate([mem_fp.ravel(), non_fp.ravel(), mem_q.ravel(), non_q.ravel()])
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < eps:
        h = np.full_like(mem_q, 0.5)
        return h, h.copy()
    s = hi - lo
    return (mem_q - lo) / s, (non_q - lo) / s


def _fp_joint_norm_diff(fp_mem: np.ndarray, fp_non: np.ndarray) -> float:
    """Matches the joint normalization in _one_cell when Q=FP: (mean(non) - mean(mem))/s,
    so the first row of the full table shares the scale of the other cells."""
    sm, snm = _joint_minmax(fp_mem, fp_non, fp_mem, fp_non)
    return _diff(snm, sm)


def _diff(nm, m):
    return float(np.mean(nm) - np.mean(m))


def _one_cell(
    cfg: dict,
    trial: int,
    device: str,
    mem,
    nm,
    fp_mem: np.ndarray | None,
    fp_non: np.ndarray | None,
    do_norm: bool,
):
    net, path = _load_net(cfg, trial, device)
    pname = _find_last_classifier_weight_name(net)
    gm = compute_grad_l2_last_layer(net, mem, device, param_name=pname)
    gnm = compute_grad_l2_last_layer(net, nm, device, param_name=pname)
    if do_norm and fp_mem is not None and fp_non is not None:
        sm, snm = _joint_minmax(fp_mem, fp_non, gm, gnm)
        d = _diff(snm, sm)
    else:
        d = _diff(gnm, gm)
    return d, {"path": path, "param": pname}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack_method", default="MIA")
    ap.add_argument("--net_name", default="resnet18")
    ap.add_argument("--dataset_name", default="cifar100")
    ap.add_argument("--random", default="42")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eps", type=float, default=0.0)
    _norm = ap.add_mutually_exclusive_group()
    _norm.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        help="Explicitly enable joint FP-Q min-max normalization (the default)",
    )
    _norm.add_argument(
        "--no_normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization and take the difference on the raw norms",
    )
    ap.set_defaults(normalize=True)
    ap.add_argument(
        "--full_table",
        action="store_true",
        help="Sweep the full table and write CSV/JSON (cells with a missing model become nan)",
    )
    ap.add_argument("--quantization", default=None, help="single-cell mode, e.g. adaround")
    ap.add_argument("--a_bit", type=int, default=None)
    ap.add_argument("--w_bit", type=int, default=None)
    ap.add_argument(
        "--out_dir",
        default=None,
        help="If given, both single-cell and full-table results are written here (no automatic subdirectories). See the module docstring for the default layout.",
    )
    ap.add_argument(
        "--no_fp_cache",
        action="store_true",
        help="Ignore any existing FP cache; recompute and overwrite the npy files under _fp_cache",
    )
    args = ap.parse_args()

    device = args.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA unavailable, using CPU")

    mem, nm = _mem_nonmem(args.dataset_name, args.random)
    print(f"members n={len(mem)}, non-members n={len(nm)}")

    do_norm = args.normalize
    base = _result_base(args.dataset_name, args.net_name)
    fp_cache = _fp_cache_dir(base, args.random, args.trial)
    mem_npy = fp_cache / "cache_fp_member_grad_l2.npy"
    non_npy = fp_cache / "cache_fp_non_member_grad_l2.npy"

    if args.full_table:
        if args.out_dir:
            out = Path(args.out_dir)
        else:
            out = base / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        out.mkdir(parents=True, exist_ok=True)
    else:
        if args.quantization is None or args.a_bit is None or args.w_bit is None:
            ap.error("single-cell mode needs --quantization --a_bit --w_bit, or use --full_table")
        if args.out_dir:
            out = Path(args.out_dir)
        else:
            out = base / f"{args.quantization}_{args.a_bit}_{args.w_bit}"
        out.mkdir(parents=True, exist_ok=True)

    print(f"result directory: {out}")
    print(f"FP cache directory: {fp_cache}")

    # ---- FP reference norms (used for joint normalization; npy files are reusable) ----
    fp_mem = fp_non = None
    cfg_fp = _cfg(
        args.attack_method,
        args.net_name,
        args.dataset_name,
        "Original",
        32,
        32,
        args.random,
        args.eps,
    )
    try:
        if (
            not args.no_fp_cache
            and mem_npy.is_file()
            and non_npy.is_file()
        ):
            fp_mem = np.load(mem_npy)
            fp_non = np.load(non_npy)
            fp_only_diff = _diff(fp_non, fp_mem)
            print(f"[FP cache] loaded {mem_npy.name} / {non_npy.name}")
            print(f"FP (32/32) raw diff (non−mem) = {fp_only_diff:.4f}")
        else:
            net_fp, _ = _load_net(cfg_fp, args.trial, device)
            pn = _find_last_classifier_weight_name(net_fp)
            fp_mem = compute_grad_l2_last_layer(net_fp, mem, device, param_name=pn)
            fp_non = compute_grad_l2_last_layer(net_fp, nm, device, param_name=pn)
            np.save(mem_npy, fp_mem)
            np.save(non_npy, fp_non)
            fp_only_diff = _diff(fp_non, fp_mem)
            print(f"FP (32/32) raw diff (non−mem) = {fp_only_diff:.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] FP not available: {e}")
        fp_only_diff = float("nan")

    if not do_norm:
        fp_mem, fp_non = None, None
    elif fp_mem is None or fp_non is None:
        print("[WARN] no FP norms available, disabling joint normalization")
        do_norm = False

    # ---- single cell ----
    if not args.full_table:
        cfg = _cfg(
            args.attack_method,
            args.net_name,
            args.dataset_name,
            args.quantization,
            args.a_bit,
            args.w_bit,
            args.random,
            args.eps,
        )
        d, meta = _one_cell(
            cfg, args.trial, device, mem, nm, fp_mem, fp_non, do_norm
        )
        row = {
            "Type": args.quantization,
            f"{args.a_bit}/{args.w_bit}": d,
            "fp_raw_diff_non_minus_mem": fp_only_diff,
            "fp_cache_dir": str(fp_cache),
            "out_dir": str(out),
            "meta": meta,
            "normalize": do_norm,
        }
        print(json.dumps(row, indent=2, ensure_ascii=False))
        out_json = out / "single_cell.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2, ensure_ascii=False)
        print(f"wrote: {out_json}")
        return

    # ---- full table ----
    table: dict[str, list] = {"Type": []}
    for b in BITS:
        table[f"{b}/{b}"] = []
    table["32/32 (FP)"] = []

    # Full-precision row: last column only
    table["Type"].append("Full-precision")
    for b in BITS:
        table[f"{b}/{b}"].append(float("nan"))
    # With normalization on, the first row's last column must share the scale of the cells below (it previously used the raw diff by mistake)
    if do_norm and fp_mem is not None and fp_non is not None:
        table["32/32 (FP)"].append(_fp_joint_norm_diff(fp_mem, fp_non))
    else:
        table["32/32 (FP)"].append(fp_only_diff)

    for lab, qname, _, _ in ROWS[1:]:
        table["Type"].append(lab)
        for b in BITS:
            cfg = _cfg(
                args.attack_method,
                args.net_name,
                args.dataset_name,
                qname,
                b,
                b,
                args.random,
                args.eps,
            )
            try:
                d, _ = _one_cell(
                    cfg, args.trial, device, mem, nm, fp_mem, fp_non, do_norm
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [skip] {lab} {b}/{b}: {e}")
                d = float("nan")
            table[f"{b}/{b}"].append(d)
        table["32/32 (FP)"].append(float("nan"))

    df = pd.DataFrame(table)
    csv_path = out / "grad_norm_diff_table.csv"
    df.to_csv(csv_path, index=False)
    meta_path = out / "run_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "normalize_joint_minmax": do_norm,
                "attack_method": args.attack_method,
                "net_name": args.net_name,
                "dataset_name": args.dataset_name,
                "random": args.random,
                "trial": args.trial,
                "out_dir": str(out),
                "fp_cache_dir": str(fp_cache),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(df.to_string(index=False))
    print(f"wrote: {csv_path}\nwrote: {meta_path}")


if __name__ == "__main__":
    main()
