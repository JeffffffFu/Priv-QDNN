#!/usr/bin/env python3
"""
Load the MIA-trained target model from save/ and compute, per sample, the L2 norm
of the cross-entropy gradient w.r.t. the last-layer classifier weights, for the
member and non-member subsets. Values and summary statistics are written to result/.

Sampling follows training/utlis.save_output and attack/WhiteBox_MIA:
  - train_data / test_data: same deterministic transform as WhiteBox (_get_eval_data)
  - target_m, shadow_m, shadow_um = split_dataset(train_data, random)
  - members: a random 20% of target_m (matching the target training subset)
  - non-members: the same number of random samples from test_data

Usage (run from the Priv-QDNN directory, same working directory as main.py):
  python extract_last_layer_grad_norm.py \\
    --attack_method MIA --net_name resnet18_drop010 --dataset_name cifar100 \\
    --quantization hawq --a_bit 8 --w_bit 8 --trial 0 --device cuda:0 --random 42

Optional: --model_path points at a .pth/.pkl directly, overriding path construction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Make project modules importable when Priv-QDNN is the cwd
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from attack.WhiteBox_MIA import _get_eval_data  # noqa: E402
from data.prepare_data import split_dataset  # noqa: E402
from training.utlis import sample_target_samples  # noqa: E402


def _unwrap_torch_model(loaded):
    """Unwrap a DNN wrapper to its .model, otherwise return the object itself."""
    return loaded.model if hasattr(loaded, "model") else loaded


def _fc_linear_like_param_names(prefix: str, module: nn.Module) -> list[str]:
    """Parameters forming the effective weight of an FC head: nn.Linear has only weight;
    BitLinear in binary mode uses pweight/nweight/scale."""
    if isinstance(module, nn.Linear):
        if module.weight is not None:
            return [f"{prefix}.weight"]
        return []
    w = getattr(module, "weight", None)
    if isinstance(w, nn.Parameter):
        return [f"{prefix}.weight"]
    names: list[str] = []
    for suf in ("pweight", "nweight", "scale"):
        p = getattr(module, suf, None)
        if isinstance(p, nn.Parameter):
            names.append(f"{prefix}.{suf}")
    return names


def _find_last_classifier_module(net: nn.Module) -> tuple[str, nn.Module] | None:
    """
    The last FC-style classifier layer in named_modules depth-first order:
    an nn.Linear, or any module with in_features/out_features that carries
    weight or pweight (e.g. BitLinear).
    """
    last: tuple[str, nn.Module] | None = None
    for name, m in net.named_modules():
        if not hasattr(m, "in_features") or not hasattr(m, "out_features"):
            continue
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                last = (name, m)
            continue
        w = getattr(m, "weight", None)
        pw = getattr(m, "pweight", None)
        if isinstance(w, nn.Parameter) or isinstance(pw, nn.Parameter):
            last = (name, m)
    return last


def _find_last_classifier_param_names(net: nn.Module) -> list[str]:
    """Names of the parameters on the last classifier head (possibly several, e.g. binary BitLinear)."""
    hit = _find_last_classifier_module(net)
    if hit is not None:
        pfx, mod = hit
        names = _fc_linear_like_param_names(pfx, mod)
        if names:
            return names
    for pname, _ in reversed(list(net.named_parameters())):
        if pname.endswith(".weight"):
            return [pname]
    raise RuntimeError("Could not find a last-layer classifier weight or any *.weight parameter in the model.")


def _find_last_classifier_weight_name(net: nn.Module) -> str:
    """
    Human-readable single string: the name itself for one parameter, comma-separated
    for several (matching how compute_grad_l2_last_layer parses it).
    """
    names = _find_last_classifier_param_names(net)
    return names[0] if len(names) == 1 else ",".join(names)


def _parse_param_names(param_name: str | list[str] | None, net: nn.Module) -> list[str]:
    if param_name is None:
        return _find_last_classifier_param_names(net)
    if isinstance(param_name, list):
        return list(param_name)
    s = param_name.strip()
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def compute_grad_l2_last_layer(
    net: nn.Module,
    subset,
    device: torch.device | str,
    param_name: str | list[str] | None = None,
) -> np.ndarray:
    """
    For each sample in the Subset, compute the L2 norm of the gradient of
    loss = CE(logits, y) w.r.t. the last-layer classifier weights.
    With several parameter names, the norm is their Euclidean combination
    sqrt(sum_i ||g_i||_2^2).
    """
    net = net.to(device)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(True)

    param_names = _parse_param_names(param_name, net)
    desc = param_names[0] if len(param_names) == 1 else f"{len(param_names)} tensors"

    criterion = nn.CrossEntropyLoss()
    norms: list[float] = []

    for i in tqdm(range(len(subset)), desc=f"grad L2 ({desc})"):
        data, label = subset[i]
        if not torch.is_tensor(data):
            data = torch.as_tensor(data)
        data = data.unsqueeze(0).to(device)
        if isinstance(label, dict):
            raise ValueError("Multi-label datasets require extending this script to name the label key.")
        label_t = torch.tensor([int(label)], dtype=torch.long, device=device)

        with torch.enable_grad():
            logits = net(data)
            loss = criterion(logits, label_t)
            net.zero_grad(set_to_none=True)
            loss.backward()

        total_sq = 0.0
        for pname in param_names:
            g = None
            for name, param in net.named_parameters():
                if name == pname:
                    g = param.grad
                    break
            if g is not None:
                total_sq += float(g.detach().float().norm(2).item() ** 2)
        norms.append(math.sqrt(total_sq) if total_sq > 0.0 else 0.0)

    return np.array(norms, dtype=np.float64)


def _default_model_path(args: dict, trial: int) -> str:
    base = os.path.join(
        os.getcwd(),
        "save",
        args["attack_method"],
        args["net_name"],
        args["dataset_name"],
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}",
    )
    if float(args.get("eps", 0) or 0) != 0.0:
        base = os.path.join(base, str(args["eps"]))

    p_trial = os.path.join(base, "target", str(trial), "model.pth")
    if os.path.isfile(p_trial):
        return p_trial
    p_flat = os.path.join(base, "target", "target_model.pkl")
    if os.path.isfile(p_flat):
        return p_flat
    raise FileNotFoundError(
        f"Model file not found. Tried:\n  {p_trial}\n  {p_flat}"
    )


def _output_dir(args: dict, trial: int) -> str:
    rel = os.path.join(
        "result",
        "last_layer_grad_norm",
        args["attack_method"],
        args["net_name"],
        args["dataset_name"],
        f"{args['quantization']}_{args['a_bit']}_{args['w_bit']}",
    )
    if float(args.get("eps", 0) or 0) != 0.0:
        rel = os.path.join(rel, str(args["eps"]))
    rel = os.path.join(rel, f"trial_{trial}")
    out = os.path.join(os.getcwd(), rel)
    os.makedirs(out, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Extract last-layer weight gradient L2 norms (members / non-members)")
    parser.add_argument("--attack_method", type=str, default="MIA")
    parser.add_argument("--net_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--quantization", type=str, required=True)
    parser.add_argument("--a_bit", type=int, required=True)
    parser.add_argument("--w_bit", type=int, required=True)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--random",
        type=str,
        default="42",
        help="Must match --random used at training time; the string 'None' means no seed",
    )
    parser.add_argument("--eps", type=float, default=0.0)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument(
        "--param_name",
        type=str,
        default=None,
        help="Parameter name(s); comma-separate several to combine their gradient L2. Defaults to the last classifier layer (nn.Linear.weight, BitLinear, ...)",
    )
    args = parser.parse_args()

    device = args.device
    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = "cpu"

    cfg = {
        "attack_method": args.attack_method,
        "net_name": args.net_name,
        "dataset_name": args.dataset_name,
        "quantization": args.quantization,
        "a_bit": args.a_bit,
        "w_bit": args.w_bit,
        "eps": args.eps,
        "random": args.random,
    }

    path = args.model_path or _default_model_path(cfg, args.trial)
    print(f"loading model: {path}")
    try:
        loaded = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        loaded = torch.load(path, map_location=device)
    net = _unwrap_torch_model(loaded)

    train_data, test_data = _get_eval_data(args.dataset_name)
    rnd = None if args.random == "None" else int(args.random)
    target_m, _shadow_m, _shadow_um = split_dataset(train_data, rnd)

    t_number = int(0.2 * len(target_m))
    member_set = sample_target_samples(target_m, t_number, args.random)
    non_member_set = sample_target_samples(test_data, t_number, args.random)

    print(f"members: {len(member_set)}, non-members: {len(non_member_set)}")

    param_name = args.param_name or _find_last_classifier_weight_name(net)
    print(f"using parameter(s): {param_name}")

    mem_norms = compute_grad_l2_last_layer(net, member_set, device, param_name=param_name)
    non_norms = compute_grad_l2_last_layer(net, non_member_set, device, param_name=param_name)

    out_dir = _output_dir(cfg, args.trial)
    np.save(os.path.join(out_dir, "member_grad_l2.npy"), mem_norms)
    np.save(os.path.join(out_dir, "non_member_grad_l2.npy"), non_norms)

    stats = {
        "param_name": param_name,
        "model_path": path,
        "trial": args.trial,
        "member_n": int(len(mem_norms)),
        "member_mean": float(np.mean(mem_norms)) if len(mem_norms) else float("nan"),
        "member_std": float(np.std(mem_norms, ddof=0)) if len(mem_norms) else float("nan"),
        "non_member_n": int(len(non_norms)),
        "non_member_mean": float(np.mean(non_norms)) if len(non_norms) else float("nan"),
        "non_member_std": float(np.std(non_norms, ddof=0)) if len(non_norms) else float("nan"),
    }
    with open(os.path.join(out_dir, "grad_norm_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    df_stats = pd.DataFrame(
        [
            ["member", stats["member_mean"], stats["member_std"], stats["member_n"]],
            ["non_member", stats["non_member_mean"], stats["non_member_std"], stats["non_member_n"]],
        ],
        columns=["group", "mean", "std", "n"],
    )
    df_stats.to_csv(os.path.join(out_dir, "grad_norm_stats.csv"), index=False)

    df_long = pd.DataFrame(
        {
            "is_member": [1] * len(mem_norms) + [0] * len(non_norms),
            "grad_l2": np.concatenate([mem_norms, non_norms]),
        }
    )
    df_long.to_csv(os.path.join(out_dir, "grad_norm_per_sample.csv"), index=False)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
