"""
MIA TPR curves (new black-box attacks + white-box attacks), read automatically from
result/*/attack_performance.csv.

Scope:
- datasets: CINIC-10, CIFAR-100
- bits: 10/8/6/4 (2-bit excluded)
- attacks: 6 new black-box (Confidence / Entropy / ModifiedEntropy / RF_m1 / SAMIA / RMIA)
           plus 2 white-box (WhiteBoxMIA / WhiteBoxMIA_Partial)

TPR metric (always TPR @ 1% FPR):
- CompLeakNR: *_tpr_at_fpr_01_mean
- RMIA: tpr_at_fpr_01_mean
- WhiteBox: WhiteBoxMIA_tpr_at_fpr_01_mean / WhiteBoxMIA_Partial_tpr_at_fpr_01_mean
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLOT_DIR.parent
RESULT_RMIA = ROOT_DIR / "result" / "RMIA" / "resnet18"
RESULT_COMP = ROOT_DIR / "result" / "MIA" / "resnet18"
RESULT_WB = ROOT_DIR / "result" / "WhiteBoxMIA" / "resnet18"
RESULT_WB_PART = ROOT_DIR / "result" / "WhiteBoxMIA_Partial" / "resnet18"
FIGURES_DIR = PLOT_DIR / "figures" / "mia_tpr"

BITS = [10, 8, 6, 4]
BIT_ORDER = [str(b) for b in BITS]
BIT_LABELS = [f"{b}/{b}" for b in BITS]

MODEL_TO_PREFIX = {
    "Pytorch-quan": "PYT_PTQ",
    "AdaRound": "adaround",
    "QDrop": "qdrop",
    "PACT": "pact",
    "HAWQ": "hawq",
    "BSQ": "bsq",
}
MODEL_ORDER = list(MODEL_TO_PREFIX.keys())

DATASET_FOLDERS = {
    "cinic10": "CINIC-10",
    "cifar100": "CIFAR-100",
}

# (key suffix, data source, CSV metric name, figure title)
ATTACK_SPECS = [
    ("Confidence", "compleak", "Confidence_tpr_at_fpr_01_mean", "Confidence"),
    ("Entropy", "compleak", "Entropy_tpr_at_fpr_01_mean", "Entropy"),
    ("ModifiedEntropy", "compleak", "ModifiedEntropy_tpr_at_fpr_01_mean", "Modified Entropy"),
    ("RF", "compleak", "train_RF_m1_tpr_at_fpr_01_mean", "RF (no LR)"),
    ("SAMIA", "compleak", "SAMIA_tpr_at_fpr_01_mean", "SAMIA"),
    ("RMIA", "rmia", "tpr_at_fpr_01_mean", "RMIA"),
    ("WhiteBox", "wb", "WhiteBoxMIA_tpr_at_fpr_01_mean", "White-box MIA"),
    ("WhiteBoxPartial", "wb_p", "WhiteBoxMIA_Partial_tpr_at_fpr_01_mean", "White-box MIA Partial"),
]


def _read_metric_csv(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("metric,") or low.startswith("metric\t"):
            continue
        parts = line.split(",", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip().lower()
        try:
            out[key] = float(parts[1].strip())
        except ValueError:
            continue
    return out


def _metric_to_percent(d: dict[str, float], metric_key: str) -> float:
    """metric_key is the lower-cased raw name from the first CSV column, e.g. samia_tpr_mean."""
    key = metric_key.lower()
    v = d.get(key)
    if v is None:
        return float("nan")
    return float(v) * 100.0 if v <= 1.0 else float(v)


def _attack_csv_path(dataset_folder: str, quant_folder: str, source: str) -> Path:
    if source == "rmia":
        base = RESULT_RMIA
    elif source == "compleak":
        base = RESULT_COMP
    elif source == "wb":
        base = RESULT_WB
    elif source == "wb_p":
        base = RESULT_WB_PART
    else:
        raise ValueError(f"unknown source: {source}")
    return base / dataset_folder / quant_folder / "attack_performance.csv"


def _build_entry(
    dataset_folder: str,
    dataset_name: str,
    source: str,
    metric_key: str,
    title_suffix: str,
) -> dict:
    """Build an entry compatible with the older plotting code: data + baseline + title."""
    models: list[str] = ["Original"]
    bit_list: list[int | str] = [32]
    values: list[float] = []

    orig_path = _attack_csv_path(dataset_folder, "Original_32_32", source)
    values.append(_metric_to_percent(_read_metric_csv(orig_path), metric_key))

    for model in MODEL_ORDER:
        prefix = MODEL_TO_PREFIX[model]
        for b in BITS:
            folder = f"{prefix}_{b}_{b}"
            p = _attack_csv_path(dataset_folder, folder, source)
            models.append(model)
            bit_list.append(b)
            values.append(_metric_to_percent(_read_metric_csv(p), metric_key))

    return {
        "data": {
            "Dataset": [dataset_name] * len(values),
            "Model": models,
            "Bit": bit_list,
            "Value": values,
        },
        "baseline": values[0],
        "title": f"{title_suffix} ({dataset_name})",
    }


def get_mia_data() -> dict:
    """Public helper: return the entries for the new black-box and white-box attacks."""
    out: dict = {}
    for ds_folder, ds_name in DATASET_FOLDERS.items():
        for key_stem, source, csv_metric, title_name in ATTACK_SPECS:
            full_key = f"MIA_{key_stem}_{ds_folder.upper()}_TPR"
            out[full_key] = _build_entry(
                ds_folder,
                ds_name,
                source,
                csv_metric,
                title_name,
            )
    return out


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"saved: {path}")
    return path


def _plot_entry(
    entry: dict,
    *,
    ylabel: str,
    save_basename: str,
    ylim: tuple[float, float] | None = None,
) -> Path:
    color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
    linestyles = ["-"] * 6
    markers = ["o", "s", "D", "^", "v", "P"]

    plot_df = pd.DataFrame(entry["data"])
    plot_df = plot_df[plot_df["Model"].isin(MODEL_ORDER)].copy()
    plot_df["Bit"] = plot_df["Bit"].astype(str)
    plot_df["Bit"] = pd.Categorical(plot_df["Bit"], categories=BIT_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, model in enumerate(MODEL_ORDER):
        subdf = plot_df[plot_df["Model"] == model]
        ax.plot(
            subdf["Bit"],
            subdf["Value"],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
        )
    ax.axhline(
        y=entry["baseline"],
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    ax.set_xlabel("Bits (W/A)", fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        vals = np.array(entry["data"]["Value"], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(np.min(vals)), float(np.max(vals))
            pad = max((hi - lo) * 0.08, 0.5)
            ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_xticks(BIT_ORDER)
    ax.set_xticklabels(BIT_LABELS, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()
    path = _save_figure(fig, save_basename)
    plt.close(fig)
    return path


def _nice_attack_name(stem: str) -> str:
    if stem == "ModifiedEntropy":
        return "modified_entropy"
    if stem == "WhiteBox":
        return "whitebox"
    if stem == "WhiteBoxPartial":
        return "whitebox_partial"
    return stem.lower()


def plot_all_mia_tpr() -> None:
    print("reading TPR from result/CompLeakNR + result/RMIA + result/WhiteBoxMIA*; saving to figures/mia_tpr/ ...")
    d = get_mia_data()
    for ds_folder in DATASET_FOLDERS.keys():
        ds_up = ds_folder.upper()
        for stem, _source, _metric, title in ATTACK_SPECS:
            key = f"MIA_{stem}_{ds_up}_TPR"
            ylabel = f"{title} TPR (%)"
            save = f"mia_tpr_{ds_folder}_{_nice_attack_name(stem)}"
            _plot_entry(d[key], ylabel=ylabel, save_basename=save, ylim=None)
    print("done.")


if __name__ == "__main__":
    plot_all_mia_tpr()
