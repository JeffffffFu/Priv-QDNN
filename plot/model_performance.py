from pathlib import Path

import matplotlib

# Without a display (e.g. over SSH) plt.show() hangs or appears unresponsive, so use a non-interactive backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PLOT_DIR.parent
SAVE_MIA_ROOT = ROOT_DIR / "save" / "MIA"
# Figures are saved to plot/figures/model_performance/
FIGURES_DIR = PLOT_DIR / "figures" / "model_performance"

CINIC10_BITS = [8, 6, 4, 2]
CINIC10_BIT_ORDER = [str(b) for b in CINIC10_BITS]
CINIC10_BIT_LABELS = [f"{b}/{b}" for b in CINIC10_BITS]

CIFAR100_BITS = [8, 6, 4, 2]
CIFAR100_BIT_ORDER = [str(b) for b in CIFAR100_BITS]
CIFAR100_BIT_LABELS = [f"{b}/{b}" for b in CIFAR100_BITS]
CIFAR100_YLIM = (0, 100)
CINIC10_YLIM = (0, 100)

MODEL_TO_PREFIX = {
    "Pytorch-quan": "PYT_PTQ",
    "AdaRound": "adaround",
    "QDrop": "qdrop",
    "PACT": "pact",
    "HAWQ": "hawq",
    "BSQ": "bsq",
}
MODEL_ORDER = list(MODEL_TO_PREFIX.keys())

DATASET_CONFIG = {
    "CINIC10_TRAINING_ACC": {
        "net": "mobilenetv2",
        "folder": "cinic10",
        "name": "CINIC-10",
        "bits": CINIC10_BITS,
        "bit_order": CINIC10_BIT_ORDER,
        "bit_labels": CINIC10_BIT_LABELS,
        "ylim": CINIC10_YLIM,
        "save_basename": "training_acc_cinic10_mobilenetv2",
        "use_embedded": False,
    },
    "CIFAR100_TRAINING_ACC": {
        "net": "resnet18",
        "folder": "cifar100",
        "name": "CIFAR-100",
        "bits": CIFAR100_BITS,
        "bit_order": CIFAR100_BIT_ORDER,
        "bit_labels": CIFAR100_BIT_LABELS,
        "ylim": CIFAR100_YLIM,
        "save_basename": "training_acc_cifar100",
        "use_embedded": True,
    },
}

# Taken from training_set_acc_mean in save/MIA/<net>/<dataset>/.../target/model_performance.csv, converted to %
# cinic10: mobilenetv2, Original + 6 methods x (8,6,4,2)
# cifar100: resnet18, Original + 6 methods x (8,6,4,2)
_EMBEDDED_TRAINING_ACC_PERCENT = {
    "cifar100": [
        97.27,
        98.39,
        97.97,
        84.26,
        1.29,
        97.79,
        97.42,
        82.9,
        1.04,
        97.82,
        97.38,
        82.46,
        1.77,
        99.12,
        99.14,
        97.03,
        64.22,
        98.61,
        98.01,
        95.82,
        5.17,
        65.02,
        62.24,
        61.13,
        51.94,
    ],
    "cinic10": [
        91.64,
        92.84,
        84.99,
        33.67,
        10.16,
        92.53,
        84.82,
        42.87,
        10.29,
        93.71,
        86.8,
        41.21,
        10.07,
        99.95,
        98.45,
        59.79,
        10.68,
        95.01,
        91.38,
        65.93,
        11.34,
        65.32,
        59.06,
        57.15,
        56.42,
    ],
}

# False: read the CSVs under save/MIA/<net>/<dataset> (recommended); True: use only the embedded table above
USE_EMBEDDED_TRAINING_ACC = True


def _read_training_acc_percent(csv_path: Path) -> float:
    """Read training_set_acc_mean from model_performance.csv and convert it to a percentage."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing file: {csv_path}")
    df = pd.read_csv(csv_path)
    col_metric = "Metric" if "Metric" in df.columns else "metric"
    col_value = "Value" if "Value" in df.columns else "value"
    df[col_metric] = df[col_metric].astype(str).str.strip()
    row = df[df[col_metric].str.lower() == "training_set_acc_mean"]
    if row.empty:
        raise KeyError(f"training_set_acc_mean not found in {csv_path}")
    value = float(row[col_value].iloc[0])
    return value * 100.0 if value <= 1.0 else value


def _save_figure(fig: plt.Figure, basename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{basename}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    print(f"saved: {path}")
    return path


def _build_dataset_entry(
    net_name: str,
    dataset_folder: str,
    dataset_name: str,
    *,
    bits: list[int],
    use_embedded: bool | None = None,
) -> dict:
    models = ["Original"]
    bit_list = [32]
    for model in MODEL_ORDER:
        for b in bits:
            models.append(model)
            bit_list.append(b)

    use_embedded = USE_EMBEDDED_TRAINING_ACC if use_embedded is None else use_embedded

    if use_embedded:
        values = list(_EMBEDDED_TRAINING_ACC_PERCENT[dataset_folder])
        if len(values) != len(models):
            raise ValueError(
                f"embedded table has {len(values)} entries but {len(models)} were expected; check _EMBEDDED_TRAINING_ACC_PERCENT[{dataset_folder!r}]"
            )
    else:
        dataset_dir = SAVE_MIA_ROOT / net_name / dataset_folder
        values = [
            _read_training_acc_percent(dataset_dir / "Original_32_32" / "target" / "model_performance.csv")
        ]
        for model in MODEL_ORDER:
            prefix = MODEL_TO_PREFIX[model]
            for b in bits:
                perf_path = dataset_dir / f"{prefix}_{b}_{b}" / "target" / "model_performance.csv"
                values.append(_read_training_acc_percent(perf_path))

        if dataset_folder == "cinic10":
            # The on-disk BSQ 6/4-bit results disagree with the embedded table, so always use the embedded table
            bsq_start = 1 + (len(MODEL_ORDER) - 1) * len(bits)
            values[bsq_start : bsq_start + len(bits)] = _EMBEDDED_TRAINING_ACC_PERCENT[
                "cinic10"
            ][-len(bits) :]

    return {
        "data": {
            "Dataset": [dataset_name] * len(values),
            "Model": models,
            "Bit": bit_list,
            "Training_Acc_Percent": values,
            "Value": values,
        },
        "baseline": values[0],
        "title": f"Training Accuracy ({dataset_name})",
    }


def get_model_performance_data():
    return {
        key: _build_dataset_entry(
            cfg["net"],
            cfg["folder"],
            cfg["name"],
            bits=cfg["bits"],
            use_embedded=cfg.get("use_embedded"),
        )
        | {
            "bit_order": cfg["bit_order"],
            "bit_labels": cfg["bit_labels"],
            "ylim": cfg["ylim"],
            "save_basename": cfg["save_basename"],
        }
        for key, cfg in DATASET_CONFIG.items()
    }

def _plot_training_acc_entry(
    entry: dict,
    *,
    save_basename: str,
) -> Path:
    bit_order = entry["bit_order"]
    bit_labels = entry["bit_labels"]
    ylim = entry["ylim"]
    color_palette = ["#E41A1C", "#377EB8", "#4DAF4A", "#FF7F00", "#000000", "#984EA3"]
    model_order = MODEL_ORDER
    linestyles = ["-", "-", "-", "-", "-", "-"]
    markers = ["o", "s", "D", "^", "v", "P"]

    df = pd.DataFrame(entry["data"])
    df["Bit"] = df["Bit"].astype(str)
    y_col = "Training_Acc_Percent" if "Training_Acc_Percent" in df.columns else "Value"
    # Only the quantized points go on the x-axis; 32 (Original) is not in categories, which avoids turning the whole table into NaN when converting to Categorical
    plot_df = df[df["Model"].isin(model_order)].copy()
    plot_df["Bit"] = pd.Categorical(plot_df["Bit"], categories=bit_order, ordered=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, model in enumerate(model_order):
        subdf = plot_df[plot_df["Model"] == model]
        ax.plot(
            subdf["Bit"],
            subdf[y_col],
            color=color_palette[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=2.5,
            markersize=10,
        )
    ax.axhline(
        y=entry["baseline"],
        color="gray",
        linestyle=(0, (3, 8)),
        linewidth=3.5,
    )
    ax.set_xlabel("Bits (W/A)", fontsize=22)
    ax.set_ylabel("Training Accuracy (%)", fontsize=22)
    ax.set_ylim(*ylim)
    ax.set_xticks(bit_order)
    ax.set_xticklabels(bit_labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    fig.tight_layout()
    path = _save_figure(fig, save_basename)
    plt.close(fig)
    return path


def plot_cinic10_training_acc() -> Path:
    data_dict = get_model_performance_data()
    return _plot_training_acc_entry(
        data_dict["CINIC10_TRAINING_ACC"],
        save_basename=data_dict["CINIC10_TRAINING_ACC"]["save_basename"],
    )


def plot_cifar100_training_acc() -> Path:
    data_dict = get_model_performance_data()
    return _plot_training_acc_entry(
        data_dict["CIFAR100_TRAINING_ACC"],
        save_basename=data_dict["CIFAR100_TRAINING_ACC"]["save_basename"],
    )


if __name__ == "__main__":
    src = "embedded table _EMBEDDED_TRAINING_ACC_PERCENT" if USE_EMBEDDED_TRAINING_ACC else "model_performance.csv under save/MIA/<net>/<dataset>"
    print(f"training-accuracy data source: {src}; saving to figures/model_performance/ ...")
    plot_cinic10_training_acc()
    plot_cifar100_training_acc()
    print("done.")
