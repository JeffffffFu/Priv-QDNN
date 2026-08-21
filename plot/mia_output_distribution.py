"""Member vs non-member MIA feature distributions before/after quantization."""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

PLOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SAVE_ROOT = PROJECT_ROOT / "save" / "MIA"
FIG_DIR = PLOT_DIR / "figures" / "mia"

BIN_EDGES = np.linspace(0.0, 1.0, 11)
BIN_WIDTH = 0.35

QUANT_METHODS: dict[str, str] = {
    "PYT_PTQ_4_4": "PQ",
    "adaround_4_4": "AdaRound",
}

DATASET_CONFIGS: dict[str, tuple[str, str, str]] = {
    "cifar100": ("resnet18", "cifar100", "CIFAR-100 ResNet18"),
    "cinic10": ("mobilenetv2", "cinic10", "CINIC-10 MobileNetV2"),
}

# AR + HAWQ @ 4-bit (C-MIA): resnet18_adversarial on CIFAR-100
DEFENSE_AR_HAWQ_CIFAR100 = {
    "net_name": "resnet18_adversarial",
    "dataset_name": "cifar100",
    "fp_dir": "Original_32_32",
    "quant_dir": "hawq_4_4",
    "fp_label": "Full-precision + AR",
    "quant_label": "HAWQ 4-bit + AR",
    "out_basename": "cifar100_ar_hawq4_output_distribution",
}

# Dropout 15% + HAWQ @ 4-bit (WB-S): resnet18_drop015 on CIFAR-100
DEFENSE_DROPOUT_WBS_CIFAR100 = {
    "net_name": "resnet18_drop015",
    "dataset_name": "cifar100",
    "fp_dir": "Original_32_32",
    "quant_dir": "hawq_4_4",
    "fp_label": "Full-precision + Dropout 15%",
    "quant_label": "HAWQ 4-bit + Dropout 15%",
    "out_basename": "cifar100_drop015_hawq4_wbs_feature_distribution",
    "random": 41,
}


def _quant_label(quant_dir: str, quant_label: str | None = None) -> str:
    if quant_label:
        return quant_label
    return QUANT_METHODS.get(quant_dir, quant_dir.replace("_4_4", ""))


def _load_uncertainties(base: Path, trial: int = 0) -> tuple[np.ndarray, np.ndarray]:
    trial_dir = base / "target" / str(trial)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    member_vals: list[float] = []
    non_member_vals: list[float] = []
    subdirs = sorted(
        (p for p in trial_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for sample_dir in subdirs:
        mem_post_path = sample_dir / "member_sample_target_model_posterior.pth"
        mem_label_path = sample_dir / "member_sample_label.pth"
        non_post_path = sample_dir / "non_member_sample_target_model_posterior.pth"
        non_label_path = sample_dir / "non_member_sample_label.pth"
        if not all(p.is_file() for p in (mem_post_path, mem_label_path, non_post_path, non_label_path)):
            continue
        mem_post = np.ravel(torch.load(mem_post_path, weights_only=False))
        mem_label = int(torch.load(mem_label_path, weights_only=False))
        non_post = np.ravel(torch.load(non_post_path, weights_only=False))
        non_label = int(torch.load(non_label_path, weights_only=False))
        member_vals.append(1.0 - float(mem_post[mem_label]))
        non_member_vals.append(1.0 - float(non_post[non_label]))
    if not member_vals:
        raise FileNotFoundError(f"No posterior samples found under {trial_dir}")
    return np.asarray(member_vals), np.asarray(non_member_vals)


def _normalized_hist(values: np.ndarray, bin_edges: np.ndarray = BIN_EDGES) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return np.zeros(len(bin_edges) - 1, dtype=float)
    return counts.astype(float) / total


def _plot_grouped_feature_hist(
    ax,
    member_vals: np.ndarray,
    non_member_vals: np.ndarray,
    *,
    bin_edges: np.ndarray,
    xlabel: str,
    show_legend: bool = True,
) -> None:
    mem_hist = _normalized_hist(member_vals, bin_edges)
    non_hist = _normalized_hist(non_member_vals, bin_edges)
    n_bins = len(mem_hist)
    x = np.arange(n_bins)
    ax.bar(
        x - BIN_WIDTH / 2,
        mem_hist,
        width=BIN_WIDTH,
        color="#E41A1C",
        edgecolor="black",
        linewidth=0.4,
        label="Members",
    )
    ax.bar(
        x + BIN_WIDTH / 2,
        non_hist,
        width=BIN_WIDTH,
        color="#084594",
        edgecolor="black",
        linewidth=0.4,
        label="Non-members",
    )
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    tick_labels = [f"{edge:.2g}" for edge in bin_edges[:-1]]
    ax.set_xticklabels(tick_labels, rotation=45 if len(tick_labels) > 8 else 0, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Normalized Frequency")
    ax.grid(True, linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend(loc="upper right")


def _plot_grouped_uncertainty_hist(ax, member_unc, non_member_unc, *, show_legend: bool = True) -> None:
    _plot_grouped_feature_hist(
        ax,
        member_unc,
        non_member_unc,
        bin_edges=BIN_EDGES,
        xlabel="Prediction Uncertainty",
        show_legend=show_legend,
    )


def _load_confidences(base: Path, trial: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Load P(true class) from saved posteriors (WB-S feature 1)."""
    trial_dir = base / "target" / str(trial)
    if not trial_dir.is_dir():
        raise FileNotFoundError(f"Missing trial directory: {trial_dir}")

    member_vals: list[float] = []
    non_member_vals: list[float] = []
    subdirs = sorted(
        (p for p in trial_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for sample_dir in subdirs:
        mem_post_path = sample_dir / "member_sample_target_model_posterior.pth"
        mem_label_path = sample_dir / "member_sample_label.pth"
        non_post_path = sample_dir / "non_member_sample_target_model_posterior.pth"
        non_label_path = sample_dir / "non_member_sample_label.pth"
        if not all(p.is_file() for p in (mem_post_path, mem_label_path, non_post_path, non_label_path)):
            continue
        mem_post = np.ravel(torch.load(mem_post_path, weights_only=False))
        mem_label = int(torch.load(mem_label_path, weights_only=False))
        non_post = np.ravel(torch.load(non_post_path, weights_only=False))
        non_label = int(torch.load(non_label_path, weights_only=False))
        member_vals.append(float(mem_post[mem_label]))
        non_member_vals.append(float(non_post[non_label]))
    if not member_vals:
        raise FileNotFoundError(f"No posterior samples found under {trial_dir}")
    return np.asarray(member_vals), np.asarray(non_member_vals)


def _parse_quant_dir(quant_dir: str) -> tuple[str, int, int]:
    quantization, a_bit, w_bit = quant_dir.rsplit("_", 2)
    return quantization, int(a_bit), int(w_bit)


def _build_whitebox_args(cfg: dict, quant_dir: str, device: str, *, trial: int = 0) -> dict:
    quantization, a_bit, w_bit = _parse_quant_dir(quant_dir)
    return {
        "net_name": cfg["net_name"],
        "dataset_name": cfg["dataset_name"],
        "quantization": quantization,
        "a_bit": a_bit,
        "w_bit": w_bit,
        "device": device,
        "random": cfg.get("random", 41),
        "trials": trial + 1,
        "eps": 0,
    }


@contextmanager
def _project_cwd():
    prev = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def _auto_bin_edges(
    member_vals: np.ndarray,
    non_member_vals: np.ndarray,
    *,
    n_bins: int = 10,
) -> np.ndarray:
    all_vals = np.concatenate([member_vals, non_member_vals])
    hi = float(np.percentile(all_vals, 99)) if len(all_vals) else 1.0
    lo = 0.0
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def _load_wbs_grad_norms(cfg: dict, quant_dir: str, *, device: str, trial: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Compute WB-S feature 2 (last-layer gradient L2 norm) for target samples."""
    from attack.WhiteBox_MIA import (
        _compute_grad_norms_matched,
        _get_eval_data,
        _reconstruct_clean_model,
        _unwrap_model,
    )
    from data.prepare_data import split_dataset
    from training.utlis import sample_target_samples

    args = _build_whitebox_args(cfg, quant_dir, device, trial=trial)
    quantization, a_bit, w_bit = _parse_quant_dir(quant_dir)
    target_path = (
        SAVE_ROOT / cfg["net_name"] / cfg["dataset_name"] / quant_dir / "target" / str(trial) / "model.pth"
    )
    if not target_path.is_file():
        raise FileNotFoundError(f"Target model not found: {target_path}")

    with _project_cwd():
        target_model_wrapped = torch.load(target_path, map_location=device, weights_only=False)
        target_net = _reconstruct_clean_model(_unwrap_model(target_model_wrapped), args, device)
        target_net.to(device).eval()

        train_data, test_data = _get_eval_data(cfg["dataset_name"])
        target_m, _, _ = split_dataset(train_data, args["random"])
        t_number = int(0.2 * len(target_m))
        t_member_set = sample_target_samples(target_m, t_number, args["random"])
        t_nonmember_set = sample_target_samples(test_data, t_number, args["random"])

        interleaved = _compute_grad_norms_matched(
            target_net, t_member_set, t_nonmember_set, device
        )

    non_member_vals = interleaved[0::2]
    member_vals = interleaved[1::2]
    return np.asarray(member_vals), np.asarray(non_member_vals)


def _load_wbs_features(
    cfg: dict,
    quant_dir: str,
    *,
    device: str,
    trial: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    base = SAVE_ROOT / cfg["net_name"] / cfg["dataset_name"] / quant_dir
    conf = _load_confidences(base, trial=trial)
    grad = _load_wbs_grad_norms(cfg, quant_dir, device=device, trial=trial)
    n = min(len(conf[0]), len(grad[0]), len(conf[1]), len(grad[1]))
    if n == 0:
        raise ValueError(f"No aligned WB-S samples for {base}")
    if n < len(conf[0]) or n < len(grad[0]):
        print(f"[WB-S plot] Aligning to {n} samples (conf={len(conf[0])}, grad={len(grad[0])})")
    return (conf[0][:n], conf[1][:n]), (grad[0][:n], grad[1][:n])


def _save_wbs_feature_figure(
    member_vals: np.ndarray,
    non_member_vals: np.ndarray,
    *,
    bin_edges: np.ndarray,
    xlabel: str,
    out_path: Path,
    show: bool,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _plot_grouped_feature_hist(
        ax,
        member_vals,
        non_member_vals,
        bin_edges=bin_edges,
        xlabel=xlabel,
    )
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    print(f"  n_member={len(member_vals)}, n_non_member={len(non_member_vals)}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def _uncertainty_base(net_name: str, dataset_name: str, quant_dir: str, *, trial: int = 0) -> Path:
    return SAVE_ROOT / net_name / dataset_name / quant_dir


def plot_defense_ar_hawq_cifar100(*, trial: int = 0, show: bool = False) -> list[Path]:
    """AR + HAWQ 4-bit vs FP+AR model output (C-MIA posterior) on CIFAR-100."""
    cfg = DEFENSE_AR_HAWQ_CIFAR100
    fp_base = _uncertainty_base(cfg["net_name"], cfg["dataset_name"], cfg["fp_dir"], trial=trial)
    q_base = _uncertainty_base(cfg["net_name"], cfg["dataset_name"], cfg["quant_dir"], trial=trial)

    fp_m, fp_n = _load_uncertainties(fp_base, trial=trial)
    q_m, q_n = _load_uncertainties(q_base, trial=trial)

    paths: list[Path] = []
    paths.append(
        _save_single_figure(
            fp_m,
            fp_n,
            out_path=FIG_DIR / f"{cfg['out_basename']}_fp.pdf",
            show=show,
        )
    )
    paths.append(
        _save_single_figure(
            q_m,
            q_n,
            out_path=FIG_DIR / f"{cfg['out_basename']}_hawq4.pdf",
            show=show,
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    panels = [
        (axes[0], fp_m, fp_n, cfg["fp_label"]),
        (axes[1], q_m, q_n, cfg["quant_label"]),
    ]
    for ax, mem_unc, non_unc, title in panels:
        _plot_grouped_uncertainty_hist(ax, mem_unc, non_unc, show_legend=(ax is axes[0]))
        ax.set_title(title, fontsize=13)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined = FIG_DIR / f"{cfg['out_basename']}.pdf"
    fig.savefig(combined, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {combined}")
    print(f"  FP+AR: n_member={len(fp_m)}, n_non_member={len(fp_n)}")
    print(f"  HAWQ4+AR: n_member={len(q_m)}, n_non_member={len(q_n)}")
    paths.append(combined)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return paths


def plot_defense_dropout_wbs_cifar100(
    *,
    trial: int = 0,
    show: bool = False,
    device: str | None = None,
    fp_dir: str | None = None,
    wbs_quant_dir: str | None = None,
) -> list[Path]:
    """Dropout 15% + HAWQ 4-bit vs FP: WB-S feature distributions on CIFAR-100."""
    cfg = dict(DEFENSE_DROPOUT_WBS_CIFAR100)
    if fp_dir is not None:
        cfg["fp_dir"] = fp_dir
    if wbs_quant_dir is not None:
        cfg["quant_dir"] = wbs_quant_dir
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    fp_base = SAVE_ROOT / cfg["net_name"] / cfg["dataset_name"] / cfg["fp_dir"]
    q_base = SAVE_ROOT / cfg["net_name"] / cfg["dataset_name"] / cfg["quant_dir"]
    fp_ready = (fp_base / "target" / str(trial)).is_dir()
    q_ready = (q_base / "target" / str(trial)).is_dir()
    if not q_ready:
        raise FileNotFoundError(f"Missing quantized data: {q_base / 'target' / str(trial)}")
    if not fp_ready:
        raise FileNotFoundError(
            f"Missing full-precision data: {fp_base / 'target' / str(trial)}. "
            "Run target training for resnet18_drop015 + Original_32_32 first."
        )

    fp_conf, fp_grad = _load_wbs_features(cfg, cfg["fp_dir"], device=device, trial=trial)
    q_conf, q_grad = _load_wbs_features(cfg, cfg["quant_dir"], device=device, trial=trial)

    fp_conf_m, fp_conf_n = fp_conf
    q_conf_m, q_conf_n = q_conf
    fp_grad_m, fp_grad_n = fp_grad
    q_grad_m, q_grad_n = q_grad

    grad_bin_edges = _auto_bin_edges(
        np.concatenate([fp_grad_m, q_grad_m]),
        np.concatenate([fp_grad_n, q_grad_n]),
    )

    paths: list[Path] = []
    basename = cfg["out_basename"]
    paths.append(
        _save_wbs_feature_figure(
            fp_conf_m,
            fp_conf_n,
            bin_edges=BIN_EDGES,
            xlabel="P(true class)",
            out_path=FIG_DIR / f"{basename}_conf_fp.pdf",
            show=show,
        )
    )
    paths.append(
        _save_wbs_feature_figure(
            q_conf_m,
            q_conf_n,
            bin_edges=BIN_EDGES,
            xlabel="P(true class)",
            out_path=FIG_DIR / f"{basename}_conf_hawq4.pdf",
            show=show,
        )
    )
    paths.append(
        _save_wbs_feature_figure(
            fp_grad_m,
            fp_grad_n,
            bin_edges=grad_bin_edges,
            xlabel="Gradient L2 Norm",
            out_path=FIG_DIR / f"{basename}_grad_fp.pdf",
            show=show,
        )
    )
    paths.append(
        _save_wbs_feature_figure(
            q_grad_m,
            q_grad_n,
            bin_edges=grad_bin_edges,
            xlabel="Gradient L2 Norm",
            out_path=FIG_DIR / f"{basename}_grad_hawq4.pdf",
            show=show,
        )
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    panels = [
        (axes[0, 0], fp_conf_m, fp_conf_n, cfg["fp_label"], BIN_EDGES, "P(true class)"),
        (axes[0, 1], q_conf_m, q_conf_n, cfg["quant_label"], BIN_EDGES, "P(true class)"),
        (axes[1, 0], fp_grad_m, fp_grad_n, cfg["fp_label"], grad_bin_edges, "Gradient L2 Norm"),
        (axes[1, 1], q_grad_m, q_grad_n, cfg["quant_label"], grad_bin_edges, "Gradient L2 Norm"),
    ]
    for ax, mem_vals, non_vals, title, bin_edges, xlabel in panels:
        _plot_grouped_feature_hist(
            ax,
            mem_vals,
            non_vals,
            bin_edges=bin_edges,
            xlabel=xlabel,
            show_legend=(ax is axes[0, 0]),
        )
        ax.set_title(title, fontsize=13)

    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    combined = FIG_DIR / f"{basename}.pdf"
    fig.savefig(combined, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {combined}")
    print(f"  FP+Drop15%: n_member={len(fp_grad_m)}, n_non_member={len(fp_grad_n)}")
    print(f"  HAWQ4+Drop15%: n_member={len(q_grad_m)}, n_non_member={len(q_grad_n)}")
    paths.append(combined)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return paths


def _save_single_figure(
    member_unc: np.ndarray,
    non_member_unc: np.ndarray,
    *,
    out_path: Path,
    show: bool,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _plot_grouped_uncertainty_hist(ax, member_unc, non_member_unc)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {out_path}")
    print(f"  n_member={len(member_unc)}, n_non_member={len(non_member_unc)}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def plot_dataset_quant(
    dataset_key: str,
    quant_dir: str,
    *,
    quant_label: str | None = None,
    trial: int = 0,
    show: bool = False,
    fp_unc: tuple[np.ndarray, np.ndarray] | None = None,
    save_fp: bool = True,
) -> list[Path]:
    """Save separate FP and 4-bit uncertainty histograms for one dataset + quant method."""
    net_name, dataset_name, _ = DATASET_CONFIGS[dataset_key]
    base = SAVE_ROOT / net_name / dataset_name

    if fp_unc is None:
        fp_m, fp_n = _load_uncertainties(base / "Original_32_32", trial=trial)
    else:
        fp_m, fp_n = fp_unc
    q_m, q_n = _load_uncertainties(base / quant_dir, trial=trial)

    paths: list[Path] = []
    if save_fp:
        paths.append(
            _save_single_figure(
                fp_m,
                fp_n,
                out_path=FIG_DIR / f"{dataset_key}_uncertainty_fp.pdf",
                show=show,
            )
        )
    paths.append(
        _save_single_figure(
            q_m,
            q_n,
            out_path=FIG_DIR / f"{dataset_key}_uncertainty_{quant_dir.lower()}.pdf",
            show=show,
        )
    )
    return paths


def plot_all(*, trial: int = 0, show: bool = False) -> list[Path]:
    """Generate separate FP + 4-bit plots for CIFAR-100 and CINIC-10 (PQ & AdaRound only)."""
    paths: list[Path] = []
    for dataset_key in DATASET_CONFIGS:
        base = SAVE_ROOT / DATASET_CONFIGS[dataset_key][0] / DATASET_CONFIGS[dataset_key][1]
        fp_unc = _load_uncertainties(base / "Original_32_32", trial=trial)
        for i, quant_dir in enumerate(QUANT_METHODS):
            quant_path = base / quant_dir
            if not quant_path.is_dir():
                print(f"Skipping missing quant dir: {quant_path}")
                continue
            paths.extend(
                plot_dataset_quant(
                    dataset_key,
                    quant_dir,
                    trial=trial,
                    show=show,
                    fp_unc=fp_unc,
                    save_fp=(i == 0),
                )
            )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MIA output uncertainty distributions.")
    parser.add_argument(
        "--mode",
        choices=("default", "defense-ar-hawq", "defense-dropout-wbs"),
        default="default",
        help=(
            "default: FP vs 4-bit PQ/AdaRound; "
            "defense-ar-hawq: AR+HAWQ on CIFAR-100; "
            "defense-dropout-wbs: Dropout15%+HAWQ WB-S on CIFAR-100."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIGS.keys()),
        default=None,
        help="Dataset to plot (default: both cifar100 and cinic10).",
    )
    parser.add_argument("--quant-dir", choices=tuple(QUANT_METHODS.keys()), default=None)
    parser.add_argument("--quant-label", default=None)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--device", default=None, help="Device for WB-S gradient computation (default: cuda:0 or cpu).")
    parser.add_argument("--wbs-fp-dir", default=None, help="Full-precision save dir name (default: Original_32_32).")
    parser.add_argument("--wbs-quant-dir", default=None, help="Quantized save dir name (default: hawq_4_4).")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    if args.mode == "defense-ar-hawq":
        plot_defense_ar_hawq_cifar100(trial=args.trial, show=args.show)
        return

    if args.mode == "defense-dropout-wbs":
        plot_defense_dropout_wbs_cifar100(
            trial=args.trial,
            show=args.show,
            device=args.device,
            fp_dir=args.wbs_fp_dir,
            wbs_quant_dir=args.wbs_quant_dir,
        )
        return

    if args.dataset is None and args.quant_dir is None:
        plot_all(trial=args.trial, show=args.show)
        return

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())
    quants = [args.quant_dir] if args.quant_dir else list(QUANT_METHODS.keys())

    for dataset_key in datasets:
        base = SAVE_ROOT / DATASET_CONFIGS[dataset_key][0] / DATASET_CONFIGS[dataset_key][1]
        fp_unc = _load_uncertainties(base / "Original_32_32", trial=args.trial)
        for i, quant_dir in enumerate(quants):
            plot_dataset_quant(
                dataset_key,
                quant_dir,
                quant_label=args.quant_label,
                trial=args.trial,
                show=args.show,
                fp_unc=fp_unc,
                save_fp=(i == 0),
            )


if __name__ == "__main__":
    main()
