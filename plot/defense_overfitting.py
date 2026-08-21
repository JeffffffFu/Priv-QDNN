"""Defense experiments: overfitting degree (same definition as in MIA_figure).

Overfitting degree = training_acc(%) - test_acc(%) in percentage points (pp).
Data comes from each backbone's ``model_performance.csv`` under ``save/MIA``
(``model_performance_dp.csv`` for DP).
Dropout rows: ``resnet18_drop010`` / ``resnet18_drop015`` each read their own directory,
falling back to ``resnet18`` when the file is missing.

Layout follows ``defense.py``: 6 defenses x 4 quantization columns, one PDF per
W/A setting (10/8/6/4).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLOT_DIR.parent
SAVE_MIA = PROJECT_ROOT / "save" / "MIA"

QUANTIZATION_METHODS = ["Full-precision", "Pytorch-quan", "AdaRound", "HAWQ"]
DEFENSE_ROWS = [
    ("W/o defense", "resnet18", ()),
    ("Adversarial learning", "resnet18_adversarial", ()),
    ("Dropout (p=0.1)", "resnet18_drop010", ()),
    ("Dropout (p=0.15)", "resnet18_drop015", ()),
    (r"DP $(\epsilon=1)$", "resnet18_dp", ("1.0",)),
    (r"DP $(\epsilon=5)$", "resnet18_dp", ("5.0",)),
]
BITWIDTHS = (10, 8, 6, 4)


def _suffix(bits: int) -> str:
    return f"{bits}_{bits}"


def _quant_folder(method: str, suf: str) -> str:
    if method == "Full-precision":
        return "Original_32_32"
    if method == "Pytorch-quan":
        return f"PYT_PTQ_{suf}"
    if method == "AdaRound":
        return f"adaround_{suf}"
    if method == "HAWQ":
        return f"hawq_{suf}"
    raise ValueError(method)


def _read_metric_csv(path: Path) -> dict[str, float]:
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
        key = parts[0].strip().lower()
        try:
            out[key] = float(parts[1].strip())
        except ValueError:
            continue
    return out


def _to_acc_percent(d: dict[str, float], key: str) -> float:
    v = d.get(key.lower())
    if v is None:
        return float("nan")
    v = float(v)
    return v * 100.0 if v <= 1.0 else v


def _perf_path(backbone: str, folder: str, eps: tuple[str, ...], dp: bool) -> Path:
    p = SAVE_MIA / backbone / "cifar100" / folder
    for e in eps:
        p = p / e
    p = p / "target"
    name = "model_performance_dp.csv" if dp else "model_performance.csv"
    return p / name


def _overfitting_from_perf(path: Path) -> float:
    d = _read_metric_csv(path)
    tr = _to_acc_percent(d, "training_set_acc_mean")
    te = _to_acc_percent(d, "test_set_acc_mean")
    if np.isnan(tr) or np.isnan(te):
        return float("nan")
    return tr - te


def _overfitting_drop_resolve(
    backbone_drop: str, folder: str, eps: tuple[str, ...]
) -> float:
    """Same as ``defense.py``: read the matching dropout backbone first, falling back to the undefended resnet18."""
    p_drop = _perf_path(backbone_drop, folder, eps, False)
    if p_drop.is_file():
        v = _overfitting_from_perf(p_drop)
        if not np.isnan(v):
            return v
    return _overfitting_from_perf(_perf_path("resnet18", folder, eps, False))


def build_overfitting_matrix(bits: int) -> np.ndarray:
    """Shape (7 defenses, 4 quant): overfitting degree (train - test, percentage points)."""
    suf = _suffix(bits)
    rows: list[list[float]] = []
    for _lab, backbone, eps in DEFENSE_ROWS:
        dp = backbone == "resnet18_dp"
        row: list[float] = []
        for q in QUANTIZATION_METHODS:
            folder = _quant_folder(q, suf)
            if backbone in ("resnet18_drop010", "resnet18_drop015"):
                row.append(_overfitting_drop_resolve(backbone, folder, eps))
            else:
                row.append(_overfitting_from_perf(_perf_path(backbone, folder, eps, dp)))
        rows.append(row)
    return np.array(rows, dtype=float)


def _save_figure(basename: str) -> None:
    out = PLOT_DIR / "figures" / "defense"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{basename}.pdf"
    plt.savefig(p, format="pdf", bbox_inches="tight")
    print(f"Saved figure: {p}")


def plot_defense_overfitting_cifar100(
    *,
    bits: int,
    data_arr: np.ndarray | None = None,
    out_basename: str | None = None,
    show: bool = False,
) -> None:
    if data_arr is None:
        data_arr = build_overfitting_matrix(bits)

    defense_methods = [d[0] for d in DEFENSE_ROWS]
    colors = ["#CC5544", "#559955", "#BEB8DC", "#9A8BD4", "#4488DD", "#DDAB44", "#997766"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    x = np.arange(len(QUANTIZATION_METHODS))
    width = 0.12
    n_def = len(defense_methods)

    for i, (_, color) in enumerate(zip(defense_methods, colors)):
        vals = np.ma.masked_invalid(data_arr[i, :])
        ax.bar(x + i * width, vals, width, color=color, alpha=0.8)

    ax.set_ylabel("Overfitting degree (%)", fontsize=36)
    # Fixed y-axis 0-80 (percentage points), matching the scale used in the paper
    ax.set_ylim(0.0, 80.0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=32)
    ax.set_xticks(x + n_def * width / 2.0)
    ax.set_xticklabels(QUANTIZATION_METHODS, fontsize=30)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0)
    y_lo, _ = ax.get_ylim()
    ax.axhline(y=y_lo, color="black", linewidth=3, alpha=0.8)

    # Defense legend: see ``plot/legend.py`` -> legend_defense_mia_confidence()
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    base = out_basename or f"defense_overfitting_cifar100_{bits}bit"
    _save_figure(base)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_all_bitwidths_defense_overfitting(show: bool = False) -> None:
    import shutil

    out_dir = PLOT_DIR / "figures" / "defense"
    for b in BITWIDTHS:
        plot_defense_overfitting_cifar100(bits=b, show=show)
    legacy = out_dir / "defense_overfitting_cifar100_10bit.pdf"
    if legacy.is_file():
        shutil.copy2(legacy, out_dir / "defense_overfitting_cifar100.pdf")
        print(f"Also saved (10-bit copy): {out_dir / 'defense_overfitting_cifar100.pdf'}")


def export_overfitting_tables() -> None:
    """Print the matrix for each bit-width, ready to paste into the paper or a constants file."""
    for b in BITWIDTHS:
        m = build_overfitting_matrix(b)
        print(f"\n# --- overfitting pp, bit {b} ---")
        for i, (lab, _, _) in enumerate(DEFENSE_ROWS):
            print(f"  {lab}: {np.round(m[i], 4).tolist()}")


if __name__ == "__main__":
    print("reading train/test from save/MIA and generating the defense overfitting figures (PDF) ...")
    plot_all_bitwidths_defense_overfitting(show=False)
    print("done.")
