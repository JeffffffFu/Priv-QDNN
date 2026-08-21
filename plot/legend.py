"""Standalone legend strips, saved to ``plot/figures/legend_*.pdf``.

The defense legend is also copied to
``plot/figures/defense/legend_defense_mia_confidence.pdf``.

Running ``python legend.py`` generates every legend PDF.
With the environment variable ``LEGEND_INTERACTIVE=1`` a window still pops up after
saving; by default the figure is only saved and closed, which suits headless environments.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

# Default to a non-interactive backend when there is no display, so saving PDFs does not fail
if os.environ.get("LEGEND_INTERACTIVE", "").lower() not in ("1", "true", "yes"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _figures_dir() -> Path:
    d = Path(__file__).resolve().parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_legend_figure(basename: str, subdir: str | None = None) -> Path:
    """Save the current figure as ``figures/[subdir/]<basename>.pdf``."""
    d = _figures_dir()
    if subdir:
        d = d / subdir
        d.mkdir(parents=True, exist_ok=True)
    path = d / f"{basename}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved legend: {path}")
    return path


def _legend_finish_interactive() -> None:
    if os.environ.get("LEGEND_INTERACTIVE", "").lower() in ("1", "true", "yes"):
        plt.show()
    else:
        plt.close("all")


def legend1():
    from matplotlib.lines import Line2D

    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    model_order = ['PQ', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(6)
    ]
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Full-precision')
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(20, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=7, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_1")
    _legend_finish_interactive()


def legend2():
    from matplotlib.lines import Line2D

    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A', '#FF7F00', '#000000', '#984EA3']
    linestyles = ['-', '-', '-', '-', '-', '-']
    markers = ['o', 's', 'D', '^', 'v', 'P']
    model_order = ['PQ', 'AdaRound', 'QDrop', 'PACT', 'HAWQ', 'BSQ']
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(6)
    ]
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Full-precision')
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=4, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_2")
    _legend_finish_interactive()


def legend3():
    from matplotlib.lines import Line2D

    color_palette = ['#E41A1C', '#377EB8', '#4DAF4A']
    linestyles = ['-', '-', '-']
    markers = ['o', 'v', 'P']
    model_order = ['PQ', 'HAWQ', 'BSQ']
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(3)
    ]
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Full-precision')
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(20, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=4, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_3")
    _legend_finish_interactive()


def legend4():
    from matplotlib.lines import Line2D

    color_palette = ['#E41A1C', '#377EB8']
    linestyles = ['-', '-']
    markers = ['o', 'o']
    model_order = ['White-box AIA', 'Black-box AIA']
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(2)
    ]
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Full-precision')
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(20, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=3, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_4")
    _legend_finish_interactive()


def legend5():
    from matplotlib.lines import Line2D

    color_palette = ['#E41A1C', '#377EB8']
    linestyles = ['-', '-']
    markers = ['o', 'o']
    model_order = ['White-box AIA', 'Black-box AIA']
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(2)
    ]
    handles.append(
        Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5, label='Full-precision')
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=3, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_5")
    _legend_finish_interactive()


def legend6():
    from matplotlib.lines import Line2D

    color_palette = [
        '#CC5544',
        '#8ECFC9',
        '#AA66CC',
        '#559955',
        '#FFBE7A',
        '#4488DD',
        '#DDAB44',
        '#997766',
    ]
    linestyles = ['-', '-', '-', '-', '-', '-', '-', '-']
    model_order = [
        'w/o defense',
        'adversarial learning (MIA)',
        'dropout',
        'adversarial learning (AIA)',
        'regularization',
        r'DP $(\epsilon=1)$',
        r'DP $(\epsilon=3)$',
        r'DP $(\epsilon=5)$',
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(8)
    ]
    labels = model_order

    fig, ax = plt.subplots(figsize=(20, 1.2))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=4, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_6")
    _legend_finish_interactive()


def legend_defense_mia_confidence():
    """Matches the defense bar colors in ``defense`` / ``defense_overfitting``; horizontal strip with ncol=4 and the same enlarged font size as the main figures."""
    import shutil
    from matplotlib.patches import Patch

    from defense_style import DEFENSE_COLORS, DEFENSE_ROW_LABELS

    colors = DEFENSE_COLORS
    labels = DEFENSE_ROW_LABELS
    handles = [
        Patch(facecolor=colors[i], edgecolor="none", alpha=1.0, label=labels[i])
        for i in range(len(labels))
    ]

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.56),
        ncol=4,
        fontsize=22,
        frameon=False,
    )
    plt.tight_layout()
    path = _save_legend_figure("legend_defense_mia_confidence")
    defense_dir = Path(__file__).resolve().parent / "figures" / "defense"
    defense_dir.mkdir(parents=True, exist_ok=True)
    defense_copy = defense_dir / "legend_defense_mia_confidence.pdf"
    shutil.copy2(path, defense_copy)
    print(f"Saved legend: {defense_copy}")
    _legend_finish_interactive()


def legend_overfitting():
    """Matches the main overfitting figure in ``MIA_figure``: 6 quantization methods + the FP baseline, on a single legend row."""
    from matplotlib.lines import Line2D

    color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
    linestyles = ["-", "-", "-", "-", "-", "-"]
    markers = ["o", "s", "D", "^", "v", "P"]
    model_order = ["PQ", "AdaRound", "QDrop", "PACT", "HAWQ", "BSQ"]
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(6)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=(0, (3, 8)),
            linewidth=3.5,
            label="Full-precision",
        )
    )
    labels = model_order + ["Full-precision"]

    fig, ax = plt.subplots(figsize=(20, 1))
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=7,
        fontsize=22,
        frameon=False,
        columnspacing=1.1,
        handlelength=2.0,
    )
    plt.tight_layout()
    _save_legend_figure("legend_overfitting", subdir="overfitting_no_legend")
    _legend_finish_interactive()


def legend_mia_cinic10_relative_attacks():
    """Matches ``MIA_figure._plot_cinic10_relative_abs_change_for_model``: a legend for the 8 attack curves."""
    from matplotlib.lines import Line2D

    # Order and colors follow ``MIA_figure.MIA_RELATIVE_*``
    colors = [
        '#E41A1C',
        '#FF7F00',
        '#984EA3',
        '#377EB8',
        '#4DAF4A',
        '#A65628',
        '#666666',
        '#66C2A5',
    ]
    labels = [
        'Confidence',
        'ML-leaks',
        'SAMIA',
        'Entropy',
        'M-Entropy',
        'RMIA',
        'WB-S',
        'WB-P',
    ]
    handles = [
        Line2D(
            [0],
            [0],
            color=colors[i],
            linestyle='-',
            marker='o',
            linewidth=2.5,
            markersize=10,
            label=labels[i],
        )
        for i in range(8)
    ]

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=4, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_mia_cinic10_relative_attacks", subdir="mia_cinic10_relative")
    _legend_finish_interactive()


def legend_mia_cinic10_relative_quantizers():
    """Matches ``MIA_figure.plot_mia_cinic10_relative_change``: 6 quantization methods + the FP baseline."""
    from matplotlib.lines import Line2D

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
    handles = [
        Line2D(
            [0],
            [0],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
            label=model_order[i],
        )
        for i in range(6)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Full-precision',
        )
    )
    labels = model_order + ['Full-precision']

    fig, ax = plt.subplots(figsize=(12, 1))
    ax.axis('off')
    ax.legend(handles=handles, labels=labels, loc='center', ncol=4, fontsize=20, frameon=False)
    plt.tight_layout()
    _save_legend_figure("legend_mia_cinic10_relative_quantizers")
    _legend_finish_interactive()


def save_all_legends() -> None:
    """Generate every legend PDF in turn (no window is shown by default)."""
    legend1()
    legend2()
    legend3()
    legend4()
    legend5()
    legend6()
    legend_defense_mia_confidence()
    legend_overfitting()
    legend_mia_cinic10_relative_attacks()
    legend_mia_cinic10_relative_quantizers()


if __name__ == "__main__":
    save_all_legends()
