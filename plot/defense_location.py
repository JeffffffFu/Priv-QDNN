"""Location (MLP-5Layer) defense-experiment MIA figure: Confidence attack + Training ACC (4-bit)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from defense_style import DEFENSE_COLORS

PLOT_DIR = Path(__file__).resolve().parent

QUANTIZATION_METHODS = ["Full-precision", "PQ", "AdaRound", "HAWQ"]
DEFENSE_ROWS = [
    ("W/o defense", "mlp_5layer", ()),
    ("Adversarial regularization", "mlp_5layer_adversarial", ()),
    (r"Drop$_{15\%}$", "mlp_5layer_drop015", ()),
    (r"Drop$_{20\%}$", "mlp_5layer_drop020", ()),
    (r"DP $(\epsilon=1)$", "mlp_5layer_dp", ("1.0",)),
    (r"DP $(\epsilon=5)$", "mlp_5layer_dp", ("5.0",)),
]

# 6 rows x 4 columns; columns = QUANTIZATION_METHODS
# W/o defense C-MIA: the main table in the paper, Location MLP (Full-precision / PQ / AdaRound / HAWQ @ 4-bit)
# C-MIA for the other defense rows: result/MIA/<backbone>/location/.../attack_performance.csv
# Training ACC: save/MIA/.../model_performance.csv (model_performance_dp.csv for DP)
PRIVACY_DATA_4BIT: list[list[float]] = [
    [73.8, 72.4, 73.4, 72.0],  # W/o defense C-MIA (TABLE IV, Location MLP @ 4-bit)
    [61.5, 63.87, 61.88, 57.38],
    [68.62, 68.25, 69.88, 69.00],
    [65.12, 67.12, 68.75, 68.00],
    [51.5, 48.62, 50.00, 50.25],
    [51.5, 50.00, 52.38, 52.38],
]

ACCURACY_DATA_4BIT: list[list[float]] = [
    [84.95, 86.19, 85.75, 84.08],
    [70.31, 68.91, 67.32, 68.91],
    [79.99, 79.99, 80.49, 84.38],
    [75.60, 78.79, 76.90, 83.73],
    [25.35, 20.01, 25.25, 20.81],
    [36.08, 35.33, 34.38, 32.39],
]


def _save_figure(basename: str) -> None:
    out = PLOT_DIR / "figures" / "defense"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{basename}.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {p}")


def plot_defense_mia_confidence_location_4bit(*, show: bool = False) -> None:
    """Two subplots: C-MIA attack ACC on top, Training ACC below (Location, 4-bit)."""
    privacy_data_arr = np.array(PRIVACY_DATA_4BIT, dtype=float)
    accuracy_data_arr = np.array(ACCURACY_DATA_4BIT, dtype=float)

    defense_methods = [d[0] for d in DEFENSE_ROWS]
    colors = DEFENSE_COLORS

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1]}
    )

    x = np.arange(len(QUANTIZATION_METHODS))
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

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, top=0.98)
    fig.canvas.draw()

    gap_mid = (ax1.get_position().y0 + ax2.get_position().y1) / 2
    middle_y = gap_mid + 0.018
    x_offsets = {
        "Full-precision": -0.01,
        "PQ": -0.02,
        "AdaRound": 0.00,
        "HAWQ": 0.02,
    }
    for i, method in enumerate(QUANTIZATION_METHODS):
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

    _save_figure("defense_mia_confidence_location_4bit")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _export_constants_from_disk() -> None:
    """Print the 4-bit constants from result/MIA and save/MIA so the embedded table can be refreshed."""
    root = PLOT_DIR.parent
    save_mia = root / "save" / "MIA"
    result_mia = root / "result" / "MIA"
    suf = "4_4"
    quants = [
        ("Full-precision", "Original_32_32"),
        ("PQ", f"PYT_PTQ_{suf}"),
        ("AdaRound", f"adaround_{suf}"),
        ("HAWQ", f"hawq_{suf}"),
    ]

    def read_csv(path: Path) -> dict[str, float]:
        if not path.is_file():
            return {}
        out: dict[str, float] = {}
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "," not in line or line.lower().startswith("metric"):
                continue
            k, v = line.split(",", 1)
            try:
                out[k.strip().lower()] = float(v.strip())
            except ValueError:
                pass
        return out

    def pct(d: dict[str, float], key: str) -> float:
        v = d.get(key)
        if v is None:
            return float("nan")
        return round(v * 100 if v <= 1 else v, 2)

    priv_rows: list[list[float]] = []
    acc_rows: list[list[float]] = []
    for _lab, backbone, eps in DEFENSE_ROWS:
        dp = backbone == "mlp_5layer_dp"
        pr_row: list[float] = []
        ac_row: list[float] = []
        for _q, folder in quants:
            ap = result_mia / backbone / "location" / folder
            sp = save_mia / backbone / "location" / folder
            for e in eps:
                ap = ap / e
                sp = sp / e
            pr_row.append(pct(read_csv(ap / "attack_performance.csv"), "confidence_acc_mean"))
            perf_name = "model_performance_dp.csv" if dp else "model_performance.csv"
            ac_row.append(pct(read_csv(sp / "target" / perf_name), "training_set_acc_mean"))
        priv_rows.append(pr_row)
        acc_rows.append(ac_row)

    print("PRIVACY_DATA_4BIT = [")
    for row in priv_rows:
        print(f"    {row},")
    print("]")
    print("ACCURACY_DATA_4BIT = [")
    for row in acc_rows:
        print(f"    {row},")
    print("]")


if __name__ == "__main__":
    plot_defense_mia_confidence_location_4bit(show=False)
