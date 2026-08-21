"""Defense-experiment MIA figure: Confidence attack only. The numbers are constants,
snapshotted from save/MIA and result/CompLeakNR, and can be edited in place."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from defense_style import DEFENSE_COLORS

PLOT_DIR = Path(__file__).resolve().parent

QUANTIZATION_METHODS = ["Full-precision", "PQ", "AdaRound", "HAWQ"]
DEFENSE_ROWS = [
    ("W/o defense", "resnet18", ()),
    ("Adversarial regularization", "resnet18_adversarial", ()),
    (r"Drop$_{15\%}$", "resnet18_drop015", ()),
    (r"Drop$_{20\%}$", "resnet18_drop020", ()),
    (r"DP $(\epsilon=1)$", "resnet18_dp", ("1.0",)),
    (r"DP $(\epsilon=5)$", "resnet18_dp", ("5.0",)),
]

BITWIDTHS = (10, 8, 6, 4)

# ---------------------------------------------------------------------------
# Constant data (6 rows x 4 columns): row order = DEFENSE_ROWS, column order = QUANTIZATION_METHODS
# - Confidence ACC (%): from result/CompLeakNR/.../attack_performance.csv
# - Training ACC (%): from save/MIA/.../model_performance.csv (model_performance_dp.csv for DP;
#   the dropout rows come from resnet18_drop020 and resnet18_drop015 respectively)
# ---------------------------------------------------------------------------
PRIVACY_DATA_BY_BIT: dict[int, list[list[float]]] = {
    10: [
        [85.24, 86.18, 86.34, 87.62],
        [52.25, 51.74, 52.57, 50.47],
        [54.55, 53.02, 53.26, 63.40],
        [74.30, 74.33, 73.79, 85.99],
        [49.99, 50.00, 50.00, 50.00],
        [49.98, 49.99, 50.01, 50.00],
    ],
    8: [
        [85.24, 86.13, 85.57, 87.31],
        [52.25, 54.15, 52.92, 51.17],
        [54.55, 56.02, 52.61, 66.15],
        [74.30, 77.50, 73.31, 87.31],
        [49.99, 50.13, 50.02, 50.00],
        [49.98, 50.02, 50.03, 50.00],
    ],
    6: [
        [85.24, 85.83, 85.19, 86.90],
        [52.25, 51.96, 53.07, 50.13],
        [54.55, 54.02, 53.78, 61.12],
        [74.30, 75.50, 72.36, 83.38],
        [49.99, 50.00, 50.02, 50.00],
        [49.98, 50.06, 50.00, float("nan")],
    ],
    # 4-bit C-MIA (TABLE IV): rows 1-5 = AR / Drop15 / Drop20 / DP eps=1 / DP eps=5; columns = FP/PQ/AdaRound/HAWQ
    4: [
        [85.20, 79.40, 80.60, 87.20],
        [53.25, 51.20, 51.65, 51.27],
        [55.55, 52.84, 52.96, 52.37],
        [50.19, 50.53, 50.43, 50.35],
        [50.31, 50.13, 50.01, 50.01],
        [51.16, 50.06, 49.85, 50.05],
    ],
}

ACCURACY_DATA_BY_BIT: dict[int, list[list[float]]] = {
    10: [
        [97.27, 98.43, 98.02, 98.50],
        [52.29, 54.63, 58.84, 49.62],
        [65.66, 64.27, 62.70, 95.64],
        [95.67, 95.80, 94.84, 98.50],
        [1.496, 1.576, 1.424, 2.480],
        [6.412, 6.932, 7.012, 9.236],
    ],
    8: [
        [97.27, 98.39, 97.79, 98.61],
        [52.29, 55.06, 58.38, 52.96],
        [65.66, 63.55, 65.12, 96.60],
        [95.67, 95.68, 95.69, 98.61],
        [1.496, 1.528, 1.404, 2.340],
        [6.412, 6.540, 6.088, 9.156],
    ],
    6: [
        [97.27, 97.97, 97.42, 98.01],
        [52.29, 53.58, 52.81, 47.77],
        [65.66, 63.89, 62.73, 95.23],
        [95.67, 95.91, 94.49, 98.01],
        [1.496, 1.764, 1.264, 2.800],
        [6.412, 6.736, 6.784, 9.204],
    ],
    4: [
        [97.27, 84.26, 82.90, 95.82],
        [52.29, 31.10, 34.19, 57.22],
        [64.56, 61.02, 43.48, 87.88],
        [36.96, 32.90, 24.16, 60.10],
        [1.496, 1.388, 1.624, 2.420],
        [6.412, 6.460, 7.052, 9.084],
    ],
}


def _save_figure(basename: str) -> None:
    out = PLOT_DIR / "figures" / "defense"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{basename}.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {p}")


def plot_defense_mia_confidence_cifar100(
    privacy_data_arr: np.ndarray | None = None,
    accuracy_data_arr: np.ndarray | None = None,
    *,
    bits: int | None = None,
    out_basename: str | None = None,
    show: bool = True,
) -> None:
    """Two subplots: Confidence ACC on top, Training ACC below (using this file's constants by default)."""
    if privacy_data_arr is None or accuracy_data_arr is None:
        if bits is None:
            bits = 10
        privacy_data_arr = np.array(PRIVACY_DATA_BY_BIT[bits], dtype=float)
        accuracy_data_arr = np.array(ACCURACY_DATA_BY_BIT[bits], dtype=float)
    else:
        bits = bits or 0

    quantization_methods = QUANTIZATION_METHODS
    defense_methods = [d[0] for d in DEFENSE_ROWS]

    colors = DEFENSE_COLORS

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1]}
    )

    x = np.arange(len(quantization_methods))
    width = 0.12

    for i, (_, color) in enumerate(zip(defense_methods, colors)):
        vals = np.ma.masked_invalid(privacy_data_arr[i, :])
        ax1.bar(x + i * width, vals, width, color=color, alpha=0.8)

    ax1.set_ylabel("Attack ACC (%)", fontsize=36)
    valid_a = privacy_data_arr[~np.isnan(privacy_data_arr)]
    if valid_a.size:
        lo = max(0.0, float(np.nanmin(valid_a)) - 5)
        hi = min(100.0, float(np.nanmax(valid_a)) + 5)
        ax1.set_ylim(lo, hi)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="y", labelsize=32)
    ax1.set_xticklabels([])
    for s in ("top", "right", "left", "bottom"):
        ax1.spines[s].set_visible(False)
    ax1.tick_params(axis="x", which="both", length=0)
    y1_lo, _ = ax1.get_ylim()
    ax1.axhline(y=y1_lo, color="black", linewidth=3, alpha=0.8)

    for i, (_, color) in enumerate(zip(defense_methods, colors)):
        vals = np.ma.masked_invalid(accuracy_data_arr[i, :])
        ax2.bar(x + i * width, vals, width, color=color, alpha=0.8)

    ax2.set_ylabel("Training ACC (%)", fontsize=36)
    valid_t = accuracy_data_arr[~np.isnan(accuracy_data_arr)]
    if valid_t.size:
        ax2.set_ylim(0, 100)
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.tick_params(axis="y", labelsize=32)
    ax2.set_xticklabels([])
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    ax2.tick_params(axis="x", which="both", length=0)
    ax2.axhline(y=0, color="black", linewidth=3, alpha=0.8)

    # The final layout must be applied before computing middle_y / fig.text, otherwise the text is offset from the subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, top=0.98)
    fig.canvas.draw()

    # Quantization method names in the gap between the two subplots (nudged relative to the gap's midpoint so they do not touch either subplot)
    gap_mid = (ax1.get_position().y0 + ax2.get_position().y1) / 2
    middle_y = gap_mid + 0.018
    x_offsets = {
        "Full-precision": -0.01,
        "PQ": -0.02,
        "AdaRound": 0.00,
        "HAWQ": 0.02,
    }
    for i, method in enumerate(quantization_methods):
        x_pos = x[i] + 2.5 * width
        x_display, _ = ax1.transData.transform((x_pos, 0))
        w = fig.get_window_extent().width
        x_norm = x_display / w if w else 0.5
        x_norm += x_offsets.get(method, 0.0)
        fig.text(
            x_norm,
            middle_y,
            method,
            ha="center",
            va="center",
            fontsize=30,
            transform=fig.transFigure,
        )

    base = out_basename or (f"defense_mia_confidence_cifar100_{bits}bit" if bits else "defense_mia_confidence_cifar100")
    _save_figure(base)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_bitwidths_cifar100(show: bool = False) -> None:
    """Generate all four figures, for 10/8/6/4 bit."""
    import shutil

    out_dir = PLOT_DIR / "figures" / "defense"
    for b in BITWIDTHS:
        plot_defense_mia_confidence_cifar100(bits=b, show=show)
    legacy = out_dir / "defense_mia_confidence_cifar100_10bit.pdf"
    if legacy.is_file():
        shutil.copy2(legacy, out_dir / "defense_mia_confidence_cifar100.pdf")
        print(f"Also saved (10-bit copy): {out_dir / 'defense_mia_confidence_cifar100.pdf'}")


# --- Optional: regenerate the constant lists from disk in JSON style (run when the project paths match) ---
def _export_constants_from_disk() -> None:
    """Call from a REPL to print the Python literals for PRIVACY_DATA_BY_BIT / ACCURACY_DATA_BY_BIT."""
    PROJECT_ROOT = PLOT_DIR.parent
    SAVE_MIA = PROJECT_ROOT / "save" / "MIA"
    RESULT_CL = PROJECT_ROOT / "result" / "CompLeakNR"

    def suffix(bits: int) -> str:
        return f"{bits}_{bits}"

    def quant_folder(method: str, suf: str) -> str:
        if method == "Full-precision":
            return "Original_32_32"
        if method == "PQ":
            return f"PYT_PTQ_{suf}"
        if method == "AdaRound":
            return f"adaround_{suf}"
        if method == "HAWQ":
            return f"hawq_{suf}"
        raise ValueError(method)

    def read_metric_csv(path: Path) -> dict[str, float]:
        if not path.is_file():
            return {}
        out: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("metric"):
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            try:
                out[parts[0].strip().lower()] = float(parts[1].strip())
            except ValueError:
                continue
        return out

    def attack_path(backbone: str, folder: str, eps: tuple[str, ...]) -> Path:
        p = RESULT_CL / backbone / "cifar100" / folder
        for e in eps:
            p = p / e
        return p / "attack_performance.csv"

    def perf_path(backbone: str, folder: str, eps: tuple[str, ...], dp: bool) -> Path:
        p = SAVE_MIA / backbone / "cifar100" / folder
        for e in eps:
            p = p / e
        p = p / "target"
        name = "model_performance_dp.csv" if dp else "model_performance.csv"
        return p / name

    def conf_pct(path: Path) -> float:
        d = read_metric_csv(path)
        v = d.get("confidence_acc_mean")
        if v is None:
            return float("nan")
        return float(v) * 100.0 if v <= 1.0 else float(v)

    def train_pct(path: Path) -> float:
        d = read_metric_csv(path)
        v = d.get("training_set_acc_mean")
        if v is None:
            return float("nan")
        return float(v) * 100.0 if v <= 1.0 else float(v)

    priv: dict[int, list[list[float]]] = {}
    acc: dict[int, list[list[float]]] = {}
    for b in BITWIDTHS:
        suf = suffix(b)
        pr_rows: list[list[float]] = []
        ac_rows: list[list[float]] = []
        for _lab, backbone, eps in DEFENSE_ROWS:
            dp = backbone == "resnet18_dp"
            pr_row = []
            ac_row = []
            for q in QUANTIZATION_METHODS:
                folder = quant_folder(q, suf)
                pr_row.append(conf_pct(attack_path(backbone, folder, eps)))
                ac_row.append(train_pct(perf_path(backbone, folder, eps, dp)))
            pr_rows.append(pr_row)
            ac_rows.append(ac_row)
        priv[b] = pr_rows
        acc[b] = ac_rows

    def fmt_block(name: str, d: dict[int, list[list[float]]]) -> None:
        print(f"{name}: dict[int, list[list[float]]] = {{")
        for b in BITWIDTHS:
            print(f"    {b}: [")
            for row in d[b]:
                print(f"        {row},")
            print("    ],")
        print("}")

    fmt_block("PRIVACY_DATA_BY_BIT", priv)
    fmt_block("ACCURACY_DATA_BY_BIT", acc)


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    plot_all_bitwidths_cifar100(show=False)
