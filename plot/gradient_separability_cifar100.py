"""
CIFAR-100 (ResNet-18) gradient-based separability - overfitting-style line plot (no legend).

Hard-coded constants from table (c); x-axis is Bits (W/A) from 8 down to 2, y-axis is
gradient-based separability; the grey dashed line is the Original baseline (0.031).
Legend: see ``legend.legend_overfitting()``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = PLOT_DIR / "figures" / "gradient_separability"

# Table (c): rows = methods, columns = bits; Original is only a horizontal baseline
TABLE: dict[str, dict[int, float]] = {
    "PQ": {2: 0.004, 4: 0.017, 6: 0.030, 8: 0.039},
    "AdaRound": {2: 0.002, 4: 0.020, 6: 0.030, 8: 0.031},
    "QDrop": {2: 0.001, 4: 0.023, 6: 0.037, 8: 0.038},
    "PACT": {2: 0.013, 4: 0.032, 6: 0.032, 8: 0.033},
    "BSQ": {2: 0.015, 4: 0.018, 6: 0.019, 8: 0.025},
    "HAWQ": {2: 0.001, 4: 0.032, 6: 0.035, 8: 0.036},
}
ORIGINAL_BASELINE = 0.031

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


def plot_gradient_separability_no_legend(*, show: bool = False) -> Path:
    """No-legend overfitting layout: 6 quantization curves plus the horizontal Original baseline."""
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
    ax.set_ylabel("Gradient-based separability", fontsize=axis_title_fs)
    ax.set_ylim(bottom=0, top=0.045)
    ax.set_xticks(list(BIT_ORDER))
    ax.set_xticklabels(BIT_LABELS, fontsize=tick_label_fs)
    ax.tick_params(axis="y", labelsize=tick_label_fs)
    fig.tight_layout()

    path = _save_figure(fig, "gradient_separability_cifar100_resnet18")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def export_table_csv() -> Path:
    rows = []
    for model in MODEL_ORDER:
        for b in (2, 4, 6, 8):
            rows.append(
                {
                    "Model": model,
                    "Bit": b,
                    "gradient_separability": TABLE[model][b],
                }
            )
    rows.append({"Model": "Original", "Bit": "", "gradient_separability": ORIGINAL_BASELINE})
    out = FIGURES_DIR / "gradient_separability_cifar100_resnet18_table.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    export_table_csv()
    plot_gradient_separability_no_legend(show=False)
    print("Done.")
