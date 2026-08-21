"""
MIA / overfitting plotting script (layout matches earlier versions).

Data policy:
- Attack AUCs live as **Python constants** in ``get_mia_data()``; nothing is read from
  disk at plotting time.
- Convention for those numbers: summarize ``attack_performance.csv`` for each method
  under ``result/`` and paste the values into this file (CompLeakNR: Confidence /
  Entropy / ModifiedEntropy / train_RF_m1 / SAMIA; RMIA: auc_mean; WhiteBoxMIA /
  WhiteBoxMIA_Partial: the corresponding columns). CSV values in 0~1 are written here
  as percentages (x100).
- The overfitting curves are also hard-coded constants, usually derived from
  ``save/MIA/.../model_performance.csv`` (unrelated to the attack files under ``result/``).

Layout: 6 black-box attacks (5 from CompLeakNR + RMIA) plus 2 white-box attacks, each
with one entry for CIFAR-100 and one for CINIC-10 (16 groups in total).
"""
from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.lines import Line2D


def _save_mia_figure(basename: str, subdir: Optional[str] = None) -> None:
    """Save current matplotlib figure to plot/figures/[subdir/]<basename>.pdf (vector)."""
    out_dir = Path(__file__).resolve().parent / "figures"
    if subdir:
        out_dir = out_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{basename}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {path}")


def _plot_overfitting_with_top_legend(
    data: dict,
    baseline: float,
    ylim_bottom: float,
    basename: str,
) -> None:
    """
    Main overfitting figure plus legend: the legend sits above the plot (outside the
    axes area) and the result is still a single PDF.
    Three quantization method names per legend row (ncol=3); the last row explains the
    full-precision reference line.
    """
    bit_order = ["8", "6", "4", "2"]
    bit_labels = ["8/8", "6/6", "4/4", "2/2"]
    axis_title_fs = 22
    tick_label_fs = 20
    df = pd.DataFrame(data)
    df["Bit"] = df["Bit"].astype(str)
    df["Bit"] = pd.Categorical(df["Bit"], categories=bit_order, ordered=True)

    color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
    model_order = ["PQ", "AdaRound", "QDrop", "PACT", "HAWQ", "BSQ"]
    linestyles = ["-", "-", "-", "-", "-", "-"]
    markers = ["o", "s", "D", "^", "v", "P"]

    fig = plt.figure(figsize=(7, 6.8))
    # Slightly shorter legend row and a small hspace; the legend is anchored to the
    # bottom of the legend axes so it sits closer to the main plot
    gs = fig.add_gridspec(2, 1, height_ratios=[0.22, 1.0], hspace=0.04)
    ax_leg = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])

    handles: list = []
    labels: list[str] = []
    for i, model in enumerate(model_order):
        subdf = df[df["Model"] == model]
        (line,) = ax_main.plot(
            subdf["Bit"],
            subdf["Value"],
            label=model,
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
        )
        handles.append(line)
        labels.append(model)

    ax_main.axhline(
        y=baseline,
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    handles.append(
        Line2D([0], [0], color="gray", linestyle=(0, (3, 8)), linewidth=3.5)
    )
    labels.append("full presicion")

    ax_leg.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
        fontsize=axis_title_fs,
        columnspacing=1.1,
        handlelength=2.0,
    )
    ax_leg.set_axis_off()

    ax_main.set_xlabel("Bits (W/A)", fontsize=axis_title_fs)
    ax_main.set_ylabel("Overfitting degree", fontsize=axis_title_fs)
    ax_main.set_ylim(bottom=ylim_bottom)
    ax_main.set_xticks(list(bit_order))
    ax_main.set_xticklabels(bit_labels, fontsize=tick_label_fs)
    ax_main.tick_params(axis="y", labelsize=tick_label_fs)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.11)
    _save_mia_figure(basename)
    plt.show()


def _plot_overfitting_no_legend(
    data: dict,
    baseline: float,
    ylim_bottom: float,
    basename: str,
    *,
    subdir: str | None = "overfitting_no_legend",
) -> None:
    """Main overfitting figure (no legend); same layout as the no-legend attack-accuracy figure."""
    bit_order = ["8", "6", "4", "2"]
    bit_labels = ["8/8", "6/6", "4/4", "2/2"]
    axis_title_fs = 22
    tick_label_fs = 20
    df = pd.DataFrame(data)
    df["Bit"] = df["Bit"].astype(str)
    df["Bit"] = pd.Categorical(df["Bit"], categories=bit_order, ordered=True)

    color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
    model_order = ["PQ", "AdaRound", "QDrop", "PACT", "HAWQ", "BSQ"]
    markers = ["o", "s", "D", "^", "v", "P"]

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, model in enumerate(model_order):
        subdf = df[df["Model"] == model]
        ax.plot(
            subdf["Bit"],
            subdf["Value"],
            color=color_palette[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
        )

    ax.axhline(
        y=baseline,
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    ax.set_xlabel("Bits (W/A)", fontsize=axis_title_fs)
    ax.set_ylabel("Overfitting degree", fontsize=axis_title_fs)
    ax.set_ylim(bottom=ylim_bottom)
    ax.set_xticks(list(bit_order))
    ax.set_xticklabels(bit_labels, fontsize=tick_label_fs)
    ax.tick_params(axis="y", labelsize=tick_label_fs)
    fig.tight_layout()
    _save_mia_figure(basename, subdir=subdir)
    plt.close(fig)


def _cifar100_overfitting_data() -> dict:
    # train−test (pp) from save/MIA/resnet18/cifar100/.../model_performance.csv; 8/6/4 are earlier results, 2 is measured at 2-bit
    return {
        "Dataset": ["CIFAR-100"] * 24,
        "Model": (
            ["PQ"] * 4
            + ["AdaRound"] * 4
            + ["QDrop"] * 4
            + ["PACT"] * 4
            + ["HAWQ"] * 4
            + ["BSQ"] * 4
        ),
        "Bit": [8, 6, 4, 2] * 6,
        "Value": [
            70.89, 70.70, 62.32, 0.05,
            70.43, 70.02, 58.61, 0.01,
            72.14, 72.40, 59.69, 0.11,
            67.46, 66.62, 66.12, 36.65,
            72.72, 72.93, 72.62, 0.74,
            48.79, 45.35, 35.80, 12.38,
        ],
    }


def _cinic10_overfitting_data() -> dict:
    return {
        "Dataset": ["CINIC-10"] * 24,
        "Model": (
            ["PQ"] * 4
            + ["AdaRound"] * 4
            + ["QDrop"] * 4
            + ["PACT"] * 4
            + ["HAWQ"] * 4
            + ["BSQ"] * 4
        ),
        "Bit": [8, 6, 4, 2] * 6,
        "Value": [
            44.98, 44.36, 33.11, 0.14,
            43.56, 43.07, 27.73, float("nan"),
            49.97, 49.57, 41.37, 1.18,
            44.07, 44.98, 42.42, 2.54,
            50.28, 49.80, 49.84, 0.16,
            32.37, 31.37, 23.71, 6.70,
        ],
    }


def _mia_frame(dataset_label: str, values_24) -> dict:
    """24 rows: 6 quantization methods x (10, 8, 6, 4) bit, fixed order PYT / AdaRound / QDrop / PACT / HAWQ / BSQ."""
    return {
        'Dataset': [dataset_label] * 24,
        'Model': (
            ['PQ'] * 4
            + ['AdaRound'] * 4
            + ['QDrop'] * 4
            + ['PACT'] * 4
            + ['HAWQ'] * 4
            + ['BSQ'] * 4
        ),
        'Bit': [10, 8, 6, 4] * 6,
        'Value': list(values_24),
    }


def get_mia_data():
    # Hard-coded constants summarized from result/.../attack_performance.csv (x100). No disk reads.
    return {
        'MIA_Confidence_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    92.11, 92.35, 91.88, 87.46,
                    92.61, 91.81, 91.65, 87.88,
                    91.89, 92.19, 92.04, 86.74,
                    92.93, 92.24, 92.51, 91.32,
                    92.62, 92.59, 92.79, 92.59,
                    79.55, 81.87, 77.72, 73.41,
                ],
            ),
            'baseline': 91.76,
            'title': 'Confidence (CIFAR-100)',
        },
        'MIA_Entropy_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    86.73, 86.67, 85.48, 73.18,
                    86.15, 85.82, 85.03, 74.31,
                    84.66, 85.67, 85.09, 71.62,
                    91.33, 88.89, 88.57, 84.83,
                    86.89, 86.84, 86.77, 86.49,
                    56.90, 60.55, 55.17, 53.27,
                ],
            ),
            'baseline': 85.02,
            'title': 'Entropy (CIFAR-100)',
        },
        'MIA_ModifiedEntropy_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    92.14, 92.39, 91.90, 87.14,
                    92.65, 91.79, 91.66, 87.51,
                    91.97, 92.34, 92.10, 86.44,
                    93.10, 92.41, 92.62, 91.35,
                    92.72, 92.70, 92.92, 92.39,
                    79.35, 81.73, 77.50, 73.46,
                ],
            ),
            'baseline': 91.69,
            'title': 'Modified Entropy (CIFAR-100)',
        },
        'MIA_RF_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    78.03, 79.41, 77.82, 66.99,
                    76.59, 76.37, 77.19, 69.71,
                    78.49, 80.25, 78.46, 67.66,
                    83.10, 81.03, 82.17, 77.96,
                    80.11, 81.01, 80.12, 81.22,
                    50.24, 51.49, 48.86, 46.22,
                ],
            ),
            'baseline': 75.98,
            'title': 'RF (no LR) (CIFAR-100)',
        },
        'MIA_SAMIA_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    92.79, 91.44, 92.25, 87.79,
                    93.06, 92.33, 92.31, 88.89,
                    93.02, 93.34, 92.98, 88.62,
                    95.18, 92.66, 92.24, 91.68,
                    94.03, 93.73, 93.83, 93.46,
                    78.59, 81.41, 75.62, 73.79,
                ],
            ),
            'baseline': 92.31,
            'title': 'SAMIA (CIFAR-100)',
        },
        'MIA_RMIA_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    93.45, 93.15, 92.41, 90.19,
                    92.93, 92.78, 92.96, 89.93,
                    93.77, 93.96, 93.89, 89.70,
                    95.05, 93.96, 94.44, 92.74,
                    92.61, 93.25, 91.95, 93.55,
                    83.72, 80.65, 85.50, 77.30,
                ],
            ),
            'baseline': 92.40,
            'title': 'RMIA (CIFAR-100)',
        },
        'MIA_WhiteBox_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    97.74, 98.15, 97.37, 89.72,
                    97.79, 97.43, 96.69, 89.94,
                    97.47, 97.84, 97.75, 91.37,
                    99.11, 97.72, 98.58, 97.19,
                    98.01, 98.02, 97.90, 97.88,
                    85.94, 89.52, 81.61, 79.74,
                ],
            ),
            'baseline': 97.25,
            'title': 'White-box MIA (CIFAR-100)',
        },
        'MIA_WhiteBoxPartial_CIFAR100': {
            'data': _mia_frame(
                'CIFAR-100',
                [
                    98.16, 98.20, 97.46, 89.83,
                    98.08, 97.78, 96.58, 89.45,
                    97.90, 98.24, 98.23, 93.82,
                    99.15, 98.77, 98.73, 98.10,
                    98.36, 98.46, 98.43, 98.06,
                    89.10, 92.23, 87.10, 85.30,
                ],
            ),
            'baseline': 97.76,
            'title': 'White-box MIA Partial (CIFAR-100)',
        },
        'MIA_Confidence_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    78.03, 75.90, 76.50, 68.82,
                    78.62, 78.00, 75.11, 68.43,
                    81.66, 81.44, 81.08, 75.59,
                    78.22, 76.57, 79.02, 77.95,
                    82.95, 82.19, 81.74, 81.09,
                    68.41, 68.59, 69.08, 64.94,
                ],
            ),
            'baseline': 78.61,
            'title': 'Confidence (CINIC-10)',
        },
        'MIA_Entropy_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    70.02, 68.33, 68.76, 58.60,
                    72.00, 69.49, 67.12, 57.11,
                    74.22, 73.54, 73.29, 64.09,
                    70.94, 69.27, 71.35, 69.84,
                    76.12, 74.95, 74.62, 72.24,
                    57.54, 57.58, 58.26, 55.68,
                ],
            ),
            'baseline': 71.43,
            'title': 'Entropy (CINIC-10)',
        },
        'MIA_ModifiedEntropy_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    78.63, 76.51, 77.03, 68.82,
                    79.28, 78.39, 75.55, 68.23,
                    82.03, 81.82, 81.54, 75.61,
                    78.46, 76.94, 79.29, 78.05,
                    83.05, 82.30, 82.15, 79.18,
                    68.32, 68.50, 69.04, 64.93,
                ],
            ),
            'baseline': 79.08,
            'title': 'Modified Entropy (CINIC-10)',
        },
        'MIA_RF_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    71.47, 69.53, 69.66, 58.08,
                    72.30, 70.70, 68.08, 57.19,
                    74.45, 73.76, 73.96, 63.76,
                    70.91, 71.03, 72.58, 70.56,
                    76.46, 75.88, 74.63, 73.49,
                    56.44, 56.73, 57.83, 54.06,
                ],
            ),
            'baseline': 73.64,
            'title': 'RF (no LR) (CINIC-10)',
        },
        'MIA_SAMIA_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    80.40, 78.79, 79.31, 70.84,
                    77.61, 80.50, 78.51, 71.51,
                    83.02, 82.57, 82.10, 77.83,
                    79.52, 79.35, 81.84, 80.80,
                    85.23, 84.85, 83.06, 81.60,
                    70.12, 69.94, 71.02, 66.35,
                ],
            ),
            'baseline': 81.69,
            'title': 'SAMIA (CINIC-10)',
        },
        'MIA_RMIA_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    81.73, 82.34, 81.61, 76.21,
                    82.89, 82.64, 83.38, 74.54,
                    85.51, 84.35, 85.94, 78.29,
                    85.61, 83.69, 83.78, 82.57,
                    87.19, 87.49, 86.68, 83.28,
                    76.54, 74.96, 71.05, 69.49,
                ],
            ),
            'baseline': 84.25,
            'title': 'RMIA (CINIC-10)',
        },
        'MIA_WhiteBox_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    92.15, 90.03, 89.25, 74.48,
                    92.62, 90.60, 87.82, 73.57,
                    93.81, 93.52, 93.34, 85.90,
                    92.12, 91.09, 92.35, 90.85,
                    93.83, 93.57, 93.98, 92.67,
                    79.86, 79.61, 79.47, 72.60,
                ],
            ),
            'baseline': 92.51,
            'title': 'White-box MIA (CINIC-10)',
        },
        'MIA_WhiteBoxPartial_CINIC10': {
            'data': _mia_frame(
                'CINIC-10',
                [
                    94.64, 93.60, 90.75, 74.96,
                    94.46, 93.22, 89.41, 73.96,
                    95.90, 95.65, 95.70, 92.98,
                    97.16, 94.98, 96.13, 95.99,
                    96.79, 96.57, 95.71, 92.98,
                    87.74, 87.63, 87.79, 84.28,
                ],
            ),
            'baseline': 94.83,
            'title': 'White-box MIA Partial (CINIC-10)',
        },
    }


# (data_dict key, output PDF basename, y-axis lower bound, y-axis upper bound); white-box CIFAR-100 goes up to 100.
MIA_SINGLE_FIGURE_SPECS = [
    ('MIA_Confidence_CIFAR100', 'mia_cifar100_confidence', 50.0, 96.0),
    ('MIA_Entropy_CIFAR100', 'mia_cifar100_entropy', 50.0, 96.0),
    ('MIA_ModifiedEntropy_CIFAR100', 'mia_cifar100_modified_entropy', 50.0, 96.0),
    ('MIA_RF_CIFAR100', 'mia_cifar100_rf', 45.0, 86.0),
    ('MIA_SAMIA_CIFAR100', 'mia_cifar100_samia', 50.0, 96.0),
    ('MIA_RMIA_CIFAR100', 'mia_cifar100_rmia', 50.0, 96.0),
    ('MIA_WhiteBox_CIFAR100', 'mia_cifar100_whitebox', 78.0, 100.0),
    ('MIA_WhiteBoxPartial_CIFAR100', 'mia_cifar100_whitebox_partial', 82.0, 100.0),
    ('MIA_Confidence_CINIC10', 'mia_cinic10_confidence', 50.0, 86.0),
    ('MIA_Entropy_CINIC10', 'mia_cinic10_entropy', 50.0, 78.0),
    ('MIA_ModifiedEntropy_CINIC10', 'mia_cinic10_modified_entropy', 50.0, 86.0),
    ('MIA_RF_CINIC10', 'mia_cinic10_rf', 50.0, 78.0),
    ('MIA_SAMIA_CINIC10', 'mia_cinic10_samia', 50.0, 88.0),
    ('MIA_RMIA_CINIC10', 'mia_cinic10_rmia', 50.0, 90.0),
    ('MIA_WhiteBox_CINIC10', 'mia_cinic10_whitebox', 70.0, 96.0),
    ('MIA_WhiteBoxPartial_CINIC10', 'mia_cinic10_whitebox_partial', 72.0, 99.0),
]

# CINIC-10: AUC change relative to the FP baseline for all 8 attacks; output directory is MIA_RELATIVE_FIGURES_SUBDIR.
MIA_RELATIVE_FIGURES_SUBDIR = "mia_cinic10_relative"
MIA_RELATIVE_ALL_KEYS_CINIC10 = (
    'MIA_Confidence_CINIC10',
    'MIA_RF_CINIC10',
    'MIA_SAMIA_CINIC10',
    'MIA_Entropy_CINIC10',
    'MIA_ModifiedEntropy_CINIC10',
    'MIA_RMIA_CINIC10',
    'MIA_WhiteBox_CINIC10',
    'MIA_WhiteBoxPartial_CINIC10',
)
MIA_RELATIVE_ALL_LABELS_CINIC10 = (
    'Confidence',
    'ML-leaks',
    'SAMIA',
    'Entropy',
    'M-Entropy',
    'RMIA',
    'WB-S',
    'WB-P',
)
# The original hex colors, one per attack in the pre-reordering order (Confidence -> ... -> WB Partial)
MIA_RELATIVE_ATTACK_COLORS = [
    '#E41A1C',
    '#FF7F00',
    '#984EA3',
    '#377EB8',
    '#4DAF4A',
    '#A65628',
    '#666666',
    '#66C2A5',
]
def _plot_mia_single_figure(
    entry_key: str,
    save_basename: str,
    y_min: float,
    y_max: float,
    data_dict: dict | None = None,
) -> None:
    """A single AUC-vs-bit curve (6 quantization methods + the FP baseline)."""
    if data_dict is None:
        data_dict = get_mia_data()
    bit_order = ['10', '8', '6', '4']
    bit_labels = ['10/10', '8/8', '6/6', '4/4']
    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    model_order = ['PQ', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    entry = data_dict[entry_key]
    df = pd.DataFrame(entry['data'])
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(7, 6))
    for i, model in enumerate(model_order):
        subdf = df[df['Model'] == model]
        plt.plot(
            subdf['Bit'],
            subdf['Value'],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
        )
    plt.axhline(
        y=entry['baseline'],
        color='gray',
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.ylim(y_min, y_max)
    plt.xticks(bit_order, bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    _save_mia_figure(save_basename)
    plt.show()


def plot_all_mia_single_figures() -> None:
    """Generate all 16 single-attack figures according to ``MIA_SINGLE_FIGURE_SPECS``."""
    dd = get_mia_data()
    for key, basename, lo, hi in MIA_SINGLE_FIGURE_SPECS:
        _plot_mia_single_figure(key, basename, lo, hi, data_dict=dd)


def _plot_cinic10_relative_abs_change_for_model(
    model_name: str,
    save_basename: str,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """One fixed quantization method, one line per attack (8 total). As in the earlier
    figures the y-axis is **AUC Change (%)**: the absolute gap between the quantized AUC
    and the FP baseline of that same attack."""
    data_dict = get_mia_data()
    bit_order = ['10', '8', '6', '4']
    bit_labels = ['10/10', '8/8', '6/6', '4/4']
    keys = MIA_RELATIVE_ALL_KEYS_CINIC10
    colors = MIA_RELATIVE_ATTACK_COLORS

    plt.figure(figsize=(7, 6))
    for attack_idx, attack_key in enumerate(keys):
        entry = data_dict[attack_key]
        baseline = entry['baseline']
        df = pd.DataFrame(entry['data'])
        sub = df[df['Model'] == model_name].copy()
        # In the categories order ['10','8',...,'4'] pandas treats later entries as larger,
        # so ascending=True is required for the x-axis to run 10/10...4/4 left to right
        # (ascending=False would put 4-bit on the far left).
        sub['Bit'] = pd.Categorical(
            sub['Bit'].astype(str), categories=bit_order, ordered=True
        )
        sub = sub.sort_values('Bit', ascending=True)
        values = sub['Value'].values
        relative_values = [abs(v - baseline) for v in values]
        plt.plot(
            bit_labels,
            relative_values,
            color=colors[attack_idx % len(colors)],
            linestyle='-',
            marker='o',
            linewidth=2.5,
            markersize=10,
        )
    plt.axhline(
        y=0, color='gray', linestyle=(0, (3, 8)), linewidth=2.5, alpha=0.7
    )
    plt.xlabel('Bits (W/A)', fontsize=22)
    plt.ylabel('AUC Change (%)', fontsize=22)
    plt.xticks(bit_labels, fontsize=20)
    plt.yticks(fontsize=20)
    # Legend: see plot/legend.py -> legend_mia_cinic10_relative_attacks()
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    _save_mia_figure(save_basename, MIA_RELATIVE_FIGURES_SUBDIR)
    plt.show()


def plot_mia_cinic10_relative_change():
    """CINIC-10: AUC change relative to the FP baseline for each attack; 8 subplots (one per
    attack), each with 6 quantization curves."""
    data_dict = get_mia_data()
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    n_attacks = len(MIA_RELATIVE_ALL_KEYS_CINIC10)
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 10), sharex=True, sharey=False)
    axes = np.asarray(axes).reshape(-1)
    plt.subplots_adjust(wspace=0.22, hspace=0.28)
    bit_order = ['10', '8', '6', '4']

    # Bright, high-contrast colors (6 quantization methods)
    color_palette = [
        '#E41A1C',
        '#377EB8',
        '#4DAF4A',
        '#FF7F00',
        '#000000',
        '#984EA3',
    ]
    model_order = ['PQ', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']

    keys = list(MIA_RELATIVE_ALL_KEYS_CINIC10)
    titles = list(MIA_RELATIVE_ALL_LABELS_CINIC10)

    for idx, key in enumerate(keys):
        ax = axes[idx]
        entry = data_dict[key]
        df = pd.DataFrame(entry['data'])
        df['Bit'] = df['Bit'].astype(str)
        df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
        
        # Change relative to the original (32-bit) model
        baseline = entry['baseline']
        relative_data = []
        
        for i, model in enumerate(model_order):
            subdf = df[df['Model'] == model]
            relative_values = subdf['Value'].values - baseline
            relative_data.append(relative_values)
            
            ax.plot(
                subdf['Bit'],
                relative_values,
                color=color_palette[i],
                linestyle=linestyles[i],
                marker=markers[i],
                linewidth=2.5,
                markersize=10,
            )

        # Zero line: the FP baseline (delta=0); legend: see legend.py -> legend_mia_cinic10_relative_quantizers()
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5)
        
        ax.set_title(f"{titles[idx]}", fontsize=22)
        ax.set_xlabel('Bits (W/A)', fontsize=22)
        ax.set_ylabel('AUC Change (%)', fontsize=22)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Set the y-axis range automatically
        all_values = np.concatenate(relative_data)
        y_margin = (all_values.max() - all_values.min()) * 0.1
        ax.set_ylim(all_values.min() - y_margin, all_values.max() + y_margin)
        
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)

    for j in range(n_attacks, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save_mia_figure("mia_cinic10_relative_change", MIA_RELATIVE_FIGURES_SUBDIR)
    plt.show()

def plot_mia_cinic10_relative_Pytorchquan():
    _plot_cinic10_relative_abs_change_for_model(
        'PQ', 'mia_cinic10_relative_pytorch_quan'
    )


def plot_mia_cinic10_relative_hawq():
    _plot_cinic10_relative_abs_change_for_model('HAWQ', 'mia_cinic10_relative_hawq')


def plot_mia_cinic10_relative_bsq():
    # The dynamic range across the 8 attacks is wide, so the y-axis is no longer fixed;
    # restore ylim if the low-AUC region needs emphasis
    _plot_cinic10_relative_abs_change_for_model('BSQ', 'mia_cinic10_relative_bsq')


def plot_mia_cinic10_relative_adaround():
    _plot_cinic10_relative_abs_change_for_model('AdaRound', 'mia_cinic10_relative_adaround')


def plot_mia_cinic10_relative_qdrop():
    _plot_cinic10_relative_abs_change_for_model('QDrop', 'mia_cinic10_relative_qdrop')


def plot_mia_cinic10_relative_pact():
    _plot_cinic10_relative_abs_change_for_model('PACT', 'mia_cinic10_relative_pact')

# === CIFAR-100 Overfitting vs Bit, main figure (MIA_MLleak_CINIC10 style) ===
def plot_overfitting_cifar100_style():
    # Overfitting: hard-coded constants. Compute (train - test) x 100 (pp) from
    # training_set_acc_mean and test_set_acc_mean in save/MIA/.../target/model_performance.csv.
    _plot_overfitting_with_top_legend(
        _cifar100_overfitting_data(), baseline=70.24, ylim_bottom=0, basename="overfitting_cifar100"
    )


def plot_overfitting_cinic10_style():
    # Same as above: hard-coded constants, computed from save/MIA/.../model_performance and pasted in.
    _plot_overfitting_with_top_legend(
        _cinic10_overfitting_data(), baseline=44.80, ylim_bottom=0, basename="overfitting_cinic10"
    )


def plot_overfitting_cifar100_no_legend():
    _plot_overfitting_no_legend(
        _cifar100_overfitting_data(),
        baseline=70.24,
        ylim_bottom=0,
        basename="overfitting_cifar100",
    )


def plot_overfitting_cinic10_no_legend():
    _plot_overfitting_no_legend(
        _cinic10_overfitting_data(),
        baseline=44.80,
        ylim_bottom=0,
        basename="overfitting_cinic10",
    )

if __name__ == "__main__":
    import os

    # Generate every figure (PDF under plot/figures/). Close figures after each to save memory.
    _ALL_PLOT_FUNCS = (
        plot_all_mia_single_figures,
        plot_mia_cinic10_relative_change,
        plot_mia_cinic10_relative_Pytorchquan,
        plot_mia_cinic10_relative_hawq,
        plot_mia_cinic10_relative_bsq,
        plot_mia_cinic10_relative_adaround,
        plot_mia_cinic10_relative_qdrop,
        plot_mia_cinic10_relative_pact,
        plot_overfitting_cifar100_style,
        plot_overfitting_cinic10_style,
        plot_overfitting_cifar100_no_legend,
        plot_overfitting_cinic10_no_legend,
    )

    # Default: do not call plt.show() 15 times (would block / open many windows). PDFs are already saved.
    # Set MIA_FIG_INTERACTIVE=1 to restore pop-up behavior for each figure.
    _interactive = os.environ.get("MIA_FIG_INTERACTIVE", "").lower() in ("1", "true", "yes")
    _orig_show = plt.show
    if not _interactive:
        plt.show = lambda *args, **kwargs: None
    try:
        for _plot_fn in _ALL_PLOT_FUNCS:
            _plot_fn()
            plt.close("all")
    finally:
        plt.show = _orig_show
