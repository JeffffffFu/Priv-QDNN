"""
CIFAR-100 (ResNet-18) model overfitting - overfitting-style line plot (no legend).

Overfitting degree = (train - test) x 100 (pp); x-axis is Bits (W/A) from 8 down to 2;
the grey dashed line is the Original baseline (70.24).
Legend: see ``legend.legend_overfitting()``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PLOT_DIR / "figures" / "overfitting_no_legend"

# 8/6/4 are earlier results; 2 is measured from save/MIA (train - test, pp)
TABLE: dict[str, dict[int, float]] = {
    "PQ": {8: 70.89, 6: 70.70, 4: 62.32, 2: 0.05},
    "AdaRound": {8: 70.43, 6: 70.02, 4: 58.61, 2: 0.01},
    "QDrop": {8: 72.14, 6: 72.40, 4: 59.69, 2: 0.11},
    "PACT": {8: 67.46, 6: 66.62, 4: 66.12, 2: 36.65},
    "HAWQ": {8: 72.72, 6: 72.93, 4: 72.62, 2: 0.74},
    "BSQ": {8: 48.79, 6: 45.35, 4: 35.80, 2: 12.38},
}
ORIGINAL_BASELINE = 70.24

BIT_ORDER = ["8", "6", "4", "2"]
BIT_LABELS = ["8/8", "6/6", "4/4", "2/2"]
MODEL_ORDER = ["PQ", "AdaRound", "QDrop", "PACT", "HAWQ", "BSQ"]
COLOR_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def _long_form_data() -> dict:
    bits_int = [8, 6, 4, 2]
    rows_model: list[str] = []
    rows_bit: list[int] = []
    rows_value: list[float] = []
    for model in MODEL_ORDER:
        for b in bits_int:
            rows_model.append(model)
            rows_bit.append(b)
            rows_value.append(TABLE[model][b])
    return {"Model": rows_model, "Bit": rows_bit, "Value": rows_value}


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def plot_overfitting_cifar100_no_legend(*, show: bool = False) -> Path:
    """Six quantization curves plus the horizontal Original baseline, no legend."""
    axis_title_fs = 22
    tick_label_fs = 20

    df = pd.DataFrame(_long_form_data())
    df["Bit"] = df["Bit"].astype(str)
    df["Bit"] = pd.Categorical(df["Bit"], categories=BIT_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, model in enumerate(MODEL_ORDER):
        subdf = df[df["Model"] == model]
        ax.plot(
            subdf["Bit"],
            subdf["Value"],
            color=COLOR_PALETTE[i],
            marker=MARKERS[i],
            linewidth=2.5,
            markersize=10,
        )

    ax.axhline(
        y=ORIGINAL_BASELINE,
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    ax.set_xlabel("Bits (W/A)", fontsize=axis_title_fs)
    ax.set_ylabel("Overfitting degree", fontsize=axis_title_fs)
    ax.set_ylim(bottom=0)
    ax.set_xticks(list(BIT_ORDER))
    ax.set_xticklabels(BIT_LABELS, fontsize=tick_label_fs)
    ax.tick_params(axis="y", labelsize=tick_label_fs)
    fig.tight_layout()

    path = _save_figure(fig, "overfitting_cifar100")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def export_table_csv() -> Path:
    rows = []
    for model in MODEL_ORDER:
        for b in (8, 6, 4, 2):
            rows.append(
                {
                    "Model": model,
                    "Bit": b,
                    "overfitting_degree_pp": TABLE[model][b],
                }
            )
    rows.append({"Model": "Original", "Bit": "", "overfitting_degree_pp": ORIGINAL_BASELINE})
    out = FIGURES_DIR / "overfitting_cifar100_resnet18_table.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    export_table_csv()
    plot_overfitting_cifar100_no_legend(show=False)
    print("Done.")
