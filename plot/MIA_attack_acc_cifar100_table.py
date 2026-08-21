"""
CIFAR-100 (ResNet-18) attack accuracy - plotted from the hard-coded constants of the
4/4 table in Figure 2 of the paper.

Layout follows ``MIA_figure._plot_overfitting_with_top_legend``:
- legend on top (ncol=2): SAMIA / WB-P
- x-axis: 7 quantization / training methods (the 4-bit quantizers plus full precision)
- y-axis: Attack Accuracy (%)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PLOT_DIR / "figures" / "mia_attack_acc"

ATTACK_LABELS = [
    "SAMIA",
    "WB-P",
]

# (mean, std) - Figure 2, 4-bit attack accuracy, in %
TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "Full-precision": {
        "SAMIA": (87.1, 0.4),
        "WB-P": (89.0, 0.9),
    },
    "PQ": {
        "SAMIA": (81.7, 0.8),
        "WB-P": (84.3, 0.8),
    },
    "AdaRound": {
        "SAMIA": (81.9, 0.3),
        "WB-P": (83.9, 0.6),
    },
    "QDrop": {
        "SAMIA": (82.1, 0.6),
        "WB-P": (85.7, 0.2),
    },
    "HAWQ": {
        "SAMIA": (85.6, 0.9),
        "WB-P": (88.5, 1.1),
    },
    "PACT": {
        "SAMIA": (86.3, 0.1),
        "WB-P": (88.3, 0.3),
    },
    "BSQ": {
        "SAMIA": (69.3, 1.2),
        "WB-P": (78.5, 1.9),
    },
}

METHOD_ORDER = [
    "Full-precision",
    "PQ",
    "AdaRound",
    "QDrop",
    "HAWQ",
    "PACT",
    "BSQ",
]
METHOD_XLABELS = [
    "Full-precision",
    "PQ",
    "AdaRound",
    "QDrop",
    "HAWQ",
    "PACT",
    "BSQ",
]

ATTACK_COLORS = ["#984EA3", "#66C2A5"]
ATTACK_MARKERS = ["P", "h"]


def _attack_series(attack: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.array([TABLE[m][attack][0] for m in METHOD_ORDER], dtype=float)
    stds = np.array([TABLE[m][attack][1] for m in METHOD_ORDER], dtype=float)
    x = np.arange(len(METHOD_ORDER))
    return x, means, stds


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def plot_attack_acc_cifar100_overfitting_style(*, show: bool = False) -> Path:
    """Main figure: 7 methods on the x-axis, SAMIA / WB-P in the legend (means only, no error bars)."""
    axis_title_fs = 22
    tick_label_fs = 20
    x = np.arange(len(METHOD_ORDER))

    fig = plt.figure(figsize=(10, 6.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.26, 1.0], hspace=0.04)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])

    handles: list = []
    labels: list[str] = []

    for i, attack in enumerate(ATTACK_LABELS):
        _, means, _ = _attack_series(attack)
        (line,) = ax_main.plot(
            x,
            means,
            color=ATTACK_COLORS[i],
            marker=ATTACK_MARKERS[i],
            linewidth=2.5,
            markersize=10,
        )
        handles.append(line)
        labels.append(attack)

    ax_leg.legend(
        handles,
        labels,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=axis_title_fs,
        columnspacing=1.0,
        handlelength=2.0,
    )
    ax_leg.set_axis_off()

    ax_main.set_xlabel("Method", fontsize=axis_title_fs)
    ax_main.set_ylabel("Attack Accuracy (%)", fontsize=axis_title_fs)
    ax_main.set_ylim(45, 95)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(METHOD_XLABELS, fontsize=tick_label_fs, rotation=25, ha="right")
    ax_main.tick_params(axis="y", labelsize=tick_label_fs)
    ax_main.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=0.20)
    path = _save_figure(fig, "attack_acc_cifar100_resnet18_overfitting_style")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def export_table_csv() -> Path:
    """Export a flat CSV for cross-checking."""
    rows = []
    for method, attacks in TABLE.items():
        for atk, (m, s) in attacks.items():
            rows.append(
                {"Method": method, "Attack": atk, "attack_accuracy_mean": m, "std": s}
            )
    out = FIGURES_DIR / "attack_acc_cifar100_resnet18_table.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    export_table_csv()
    plot_attack_acc_cifar100_overfitting_style(show=False)
    print("Done.")
