"""
CINIC-10 (MobileNetV2) model overfitting - overfitting-style line plot (no legend).

Overfitting degree = (train - test) x 100 (pp); x-axis is Bits (W/A) from 8 down to 2;
the grey dashed line is the Original baseline. Data comes from
save/MIA/mobilenetv2/cinic10/.../target/model_performance.csv.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLOT_DIR.parent
SAVE_MIA_DIR = ROOT_DIR / "save" / "MIA" / "mobilenetv2" / "cinic10"
FIGURES_DIR = PLOT_DIR / "figures" / "overfitting_no_legend"

MODEL_TO_PREFIX = {
    "PQ": "PYT_PTQ",
    "AdaRound": "adaround",
    "QDrop": "qdrop",
    "PACT": "pact",
    "HAWQ": "hawq",
    "BSQ": "bsq",
}
MODEL_ORDER = list(MODEL_TO_PREFIX.keys())
BITS = [8, 6, 4, 2]
BIT_ORDER = [str(b) for b in BITS]
BIT_LABELS = [f"{b}/{b}" for b in BITS]

COLOR_PALETTE = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
MARKERS = ["o", "s", "D", "^", "v", "P"]

# Overfitting measured from save/MIA (train - test, pp); the BSQ 6/4-bit values match the embedded table in model_performance
_EMBEDDED_OVERFITTING_PP = {
    "PQ": {8: 52.63, 6: 45.99, 4: 5.85, 2: 0.27},
    "AdaRound": {8: 51.67, 6: 46.35, 4: 11.64, 2: 0.20},
    "QDrop": {8: 53.76, 6: 47.44, 4: 10.03, 2: 0.02},
    "PACT": {8: 58.48, 6: 57.31, 4: 17.79, 2: 0.18},
    "HAWQ": {8: 54.18, 6: 50.02, 4: 19.47, 2: 1.26},
    "BSQ": {8: 28.66, 6: 22.19, 4: 21.86, 2: 9.39},
}
ORIGINAL_BASELINE = 50.70

USE_EMBEDDED = True


def _read_overfitting_pp(csv_path: Path) -> float:
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing file: {csv_path}")
    train = test = None
    for line in csv_path.read_text().splitlines():
        if "," not in line:
            continue
        k, v = line.split(",", 1)
        key = k.strip().lower()
        if key == "training_set_acc_mean":
            train = float(v.strip())
        elif key == "test_set_acc_mean":
            test = float(v.strip())
    if train is None or test is None:
        raise KeyError(f"{csv_path} is missing training/test acc")
    train_pct = train * 100 if train <= 1 else train
    test_pct = test * 100 if test <= 1 else test
    return round(train_pct - test_pct, 2)


def _build_table() -> dict[str, dict[int, float]]:
    if USE_EMBEDDED:
        return {k: dict(v) for k, v in _EMBEDDED_OVERFITTING_PP.items()}

    table: dict[str, dict[int, float]] = {}
    for model, prefix in MODEL_TO_PREFIX.items():
        table[model] = {}
        for b in BITS:
            folder = f"{prefix}_{b}_{b}"
            table[model][b] = _read_overfitting_pp(
                SAVE_MIA_DIR / folder / "target" / "model_performance.csv"
            )
    return table


def _original_baseline() -> float:
    if USE_EMBEDDED:
        return ORIGINAL_BASELINE
    return _read_overfitting_pp(SAVE_MIA_DIR / "Original_32_32" / "target" / "model_performance.csv")


def _long_form_data(table: dict[str, dict[int, float]]) -> dict:
    rows_model: list[str] = []
    rows_bit: list[int] = []
    rows_value: list[float] = []
    for model in MODEL_ORDER:
        for b in BITS:
            rows_model.append(model)
            rows_bit.append(b)
            rows_value.append(table[model][b])
    return {"Model": rows_model, "Bit": rows_bit, "Value": rows_value}


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def plot_overfitting_cinic10_mobilenetv2_no_legend(*, show: bool = False) -> Path:
    table = _build_table()
    baseline = _original_baseline()

    df = pd.DataFrame(_long_form_data(table))
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

    ax.axhline(y=baseline, color="gray", linestyle=(0, (3, 8)), linewidth=3.5)
    ax.set_xlabel("Bits (W/A)", fontsize=22)
    ax.set_ylabel("Overfitting degree", fontsize=22)
    ax.set_ylim(bottom=0)
    ax.set_xticks(BIT_ORDER)
    ax.set_xticklabels(BIT_LABELS, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()

    path = _save_figure(fig, "overfitting_cinic10_mobilenetv2")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def export_table_csv() -> Path:
    table = _build_table()
    baseline = _original_baseline()
    rows = []
    for model in MODEL_ORDER:
        for b in BITS:
            rows.append(
                {
                    "Model": model,
                    "Bit": b,
                    "overfitting_degree_pp": table[model][b],
                }
            )
    rows.append({"Model": "Original", "Bit": "", "overfitting_degree_pp": baseline})
    out = FIGURES_DIR / "overfitting_cinic10_mobilenetv2_table.csv"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved: {out}")
    return out


if __name__ == "__main__":
    export_table_csv()
    plot_overfitting_cinic10_mobilenetv2_no_legend(show=False)
    print("Done.")
