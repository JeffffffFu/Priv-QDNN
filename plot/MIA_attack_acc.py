"""
CIFAR-100 (ResNet-18) attack-accuracy curves - SAMIA / WB-P.

Layout follows the main overfitting figure (no legend):
- one PDF per attack (2 in total)
- x-axis: Bits (8/8, 6/6, 4/4, 2/2)
- curves: 6 quantization methods (PQ, AdaRound, QDrop, PACT, HAWQ, BSQ)
- grey dashed line: the full-precision (32-bit) baseline
- 8/6-bit: hard-coded constants from the summary table in Figure 1 of the paper
- 4-bit: hard-coded constants from the table in Figure 2 of the paper
- 2-bit: measured from the experiment results
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PLOT_DIR / "figures" / "mia_attack_acc_by_bits"

BIT_ORDER = ["8", "6", "4", "2"]
BIT_LABELS = ["8/8", "6/6", "4/4", "2/2"]
MODEL_ORDER = ["PQ", "AdaRound", "QDrop", "PACT", "HAWQ", "BSQ"]

COLOR_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

# CIFAR-100 - ResNet-18 - Figure 1 (8/8, 6/6) - Attack Accuracy, %
EIGHT_BIT_TABLE: dict[str, dict[str, float]] = {
    "SAMIA": {
        "PQ": 87.8,
        "AdaRound": 87.3,
        "QDrop": 88.4,
        "PACT": 88.9,
        "HAWQ": 88.7,
        "BSQ": 77.2,
    },
    "WB-P": {
        "PQ": 89.7,
        "AdaRound": 88.6,
        "QDrop": 90.1,
        "PACT": 90.2,
        "HAWQ": 91.0,
        "BSQ": 84.1,
    },
}

SIX_BIT_TABLE: dict[str, dict[str, float]] = {
    "SAMIA": {
        "PQ": 87.6,
        "AdaRound": 87.2,
        "QDrop": 87.7,
        "PACT": 87.1,
        "HAWQ": 89.0,
        "BSQ": 71.4,
    },
    "WB-P": {
        "PQ": 89.1,
        "AdaRound": 88.1,
        "QDrop": 90.2,
        "PACT": 90.6,
        "HAWQ": 90.9,
        "BSQ": 80.0,
    },
}

# CIFAR-100 - ResNet-18 - Figure 2 (4/4) - Attack Accuracy, %
FOUR_BIT_TABLE: dict[str, dict[str, float]] = {
    "SAMIA": {
        "Full-precision": 87.1,
        "PQ": 81.7,
        "AdaRound": 81.9,
        "QDrop": 82.1,
        "HAWQ": 85.6,
        "PACT": 86.3,
        "BSQ": 69.3,
    },
    "WB-P": {
        "Full-precision": 89.0,
        "PQ": 84.3,
        "AdaRound": 83.9,
        "QDrop": 85.7,
        "HAWQ": 88.5,
        "PACT": 88.3,
        "BSQ": 78.5,
    },
}

TWO_BIT_TABLE: dict[str, dict[str, float]] = {
    "SAMIA": {
        "PQ": 51.9,
        "AdaRound": 52.2,
        "QDrop": 51.9,
        "PACT": 70.0,
        "HAWQ": 50.5,
        "BSQ": 58.1,
    },
    "WB-P": {
        "PQ": 57.7,
        "AdaRound": 53.6,
        "QDrop": 52.6,
        "PACT": 80.6,
        "HAWQ": 49.3,
        "BSQ": 65.9,
    },
}

FP_BASELINE = {"SAMIA": 87.1, "WB-P": 89.0}


def _bit_series(attack: str) -> list[float]:
    """Six methods x (8, 6, 4, 2) bit."""
    values: list[float] = []
    for model in MODEL_ORDER:
        values.extend([
            EIGHT_BIT_TABLE[attack][model],
            SIX_BIT_TABLE[attack][model],
            FOUR_BIT_TABLE[attack][model],
            TWO_BIT_TABLE[attack][model],
        ])
    return values


# attack -> (FP baseline, 24 means)
ATTACK_SPECS: dict[str, tuple[float, list[float | None]]] = {
    attack: (FP_BASELINE[attack], _bit_series(attack))
    for attack in ("SAMIA", "WB-P")
}


def _attack_frame(values_24: list[float | None]) -> dict:
    return {
        "Model": (
            ["PQ"] * 4
            + ["AdaRound"] * 4
            + ["QDrop"] * 4
            + ["PACT"] * 4
            + ["HAWQ"] * 4
            + ["BSQ"] * 4
        ),
        "Bit": [8, 6, 4, 2] * 6,
        "Value": list(values_24),
    }


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def _auto_ylim(values: list[float | None], baseline: float) -> tuple[float, float]:
    numeric = [v for v in values if v is not None] + [baseline]
    all_v = np.array(numeric, dtype=float)
    lo, hi = float(np.min(all_v)), float(np.max(all_v))
    pad = max((hi - lo) * 0.08, 2.0)
    return max(0.0, lo - pad), hi + pad


def plot_attack_acc_single(
    attack: str,
    *,
    ylim: tuple[float, float] | None = None,
    show: bool = False,
) -> Path:
    """A single attack-accuracy vs bit curve (no legend)."""
    baseline, values_24 = ATTACK_SPECS[attack]
    df = pd.DataFrame(_attack_frame(values_24))
    df["Bit"] = df["Bit"].astype(str)
    df["Bit"] = pd.Categorical(df["Bit"], categories=BIT_ORDER, ordered=True)

    axis_title_fs = 22
    tick_label_fs = 20
    if ylim is None:
        ylim = _auto_ylim(values_24, baseline)

    fig, ax_main = plt.subplots(figsize=(7, 6))

    for i, model in enumerate(MODEL_ORDER):
        subdf = df[df["Model"] == model].dropna(subset=["Value"])
        if subdf.empty:
            continue
        ax_main.plot(
            subdf["Bit"],
            subdf["Value"],
            color=COLOR_PALETTE[i],
            marker=MARKERS[i],
            linewidth=2.5,
            markersize=10,
        )

    ax_main.axhline(
        y=baseline,
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )

    ax_main.set_xlabel("Bits (W/A)", fontsize=axis_title_fs)
    ax_main.set_ylabel("Attack Accuracy (%)", fontsize=axis_title_fs)
    ax_main.set_ylim(*ylim)
    ax_main.set_xticks(BIT_ORDER)
    ax_main.set_xticklabels(BIT_LABELS, fontsize=tick_label_fs)
    ax_main.tick_params(axis="y", labelsize=tick_label_fs)

    fig.tight_layout()
    slug = attack.lower().replace("-", "_")
    path = _save_figure(fig, f"attack_acc_cifar100_{slug}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_all_attack_acc(*, show: bool = False) -> None:
    """Generate the accuracy-curve PDFs for SAMIA and WB-P."""
    for attack in ATTACK_SPECS:
        plot_attack_acc_single(attack, show=show)


def export_table_csv() -> Path:
    """Export a flat CSV (means only)."""
    rows = []
    for attack, (fp, values) in ATTACK_SPECS.items():
        rows.append(
            {"Method": "Full-precision", "Bits": 32, "Attack": attack, "Value": fp}
        )
        frame = _attack_frame(values)
        for model, bit, val in zip(frame["Model"], frame["Bit"], frame["Value"]):
            rows.append({"Method": model, "Bits": bit, "Attack": attack, "Value": val})
    out = FIGURES_DIR / "attack_acc_cifar100_table.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    export_table_csv()
    plot_all_attack_acc(show=False)
    print("Done.")
