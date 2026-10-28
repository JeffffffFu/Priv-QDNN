# Priv-QDNN
This is the offical code of Paper <<Revisiting Privacy Risks in Quantized Deep Neural Networks>>
This repository contains a research codebase for **privacy evaluation of quantized deep neural networks**. It supports training target/shadow models under common post-training quantization (PTQ) and quantization-aware training (QAT) settings, saving intermediate attack features to disk, and running a suite of **membership-inference-related** attacks and metrics (e.g., `MIA`, `RMIA`, `WhiteBoxMIA`, `WhiteBoxMIA_Partial`). The codebase also integrates **Opacus**-based differential privacy training paths for defended models (e.g., dropout / DP / adversarial variants in the provided example scripts).

The evaluation workflow is split into two stages: **(1) training** (`--training target|shadow|both`) to materialize `save/<attack_method>/...` artifacts, and **(2) attack-only evaluation** (`--training None`) that loads saved posteriors/models and reports metrics under `result/<attack_method>/...`.

## Installation

You need a machine with **NVIDIA GPU** support for typical experiments. We recommend **CUDA 12.1** wheels for PyTorch (see commands below). Use **Python 3.10** to match the versions pinned in `requirements.txt`.

You can install most requirements with:

```bash
conda create --name priv-qdnn python=3.10
conda activate priv-qdnn
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Notes:

- For **CPU-only** PyTorch, run `pip install -r requirements.txt` without `--extra-index-url` (PyPI CPU wheels if available for your platform).
- If you use **AdaRound / AIMET** quantization paths, you need a working `aimet-torch` install (see `quantization/PTQ/adaround.py`). If `pip` cannot resolve `aimet-torch` on your platform, follow Qualcomm AIMET installation instructions first, then install the remaining dependencies.

## Datasets

Datasets are **not** included in this repository. Four are supported, and they differ in how much setup they need.

### CIFAR-100 and CINIC-10 (automatic)

Both are fetched on first use into `data/`, so no manual step is required — just make sure the machine has network access the first time you run an experiment.

- **CIFAR-100** is downloaded by `torchvision.datasets.CIFAR100(..., download=True)` in `data/load_data.py`.
- **CINIC-10** is downloaded and extracted by `fetch_cinic10()` in `data/cinic10.py` from the Edinburgh DataShare mirror. It is a large archive, so budget time and disk for the first run.

### Texas100 and Location (manual)

These two have no automatic downloader. Fetch the raw archives, unpack them under `data/<dataset>/raw/`, then convert them to the `.npy` files the loaders expect.

| Dataset | Raw archive |
| --- | --- |
| Texas100 | <https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz> |
| Location | <https://github.com/privacytrustlab/datasets> (the `location` set) |

Expected layout after unpacking:

```
data/texas/raw/texas/100/feats
data/texas/raw/texas/100/labels
data/location/raw/bangkok
```

Then build the train/test splits:

```bash
python data/prepare_tabular.py --dataset both
```

`--dataset` accepts `texas`, `location`, or `both`. Use `--raw_root` to point at a different location (default `data/<dataset>/raw`), and `--test_size` / `--seed` to change the split (defaults `0.2` and `41`).

This writes the files that `data/tabular_datasets.py` loads:

```
data/<dataset>/{train_feats,train_labels,test_feats,test_labels}.npy
```

If these `.npy` files are missing, any run using `--dataset_name texas` or `--dataset_name location` fails with a `FileNotFoundError` that names the missing path.

## Code Structure of Priv-QDNN

- **`attack`**: attack implementations and evaluation entry points (`MIA`, `RMIA`, `WhiteBox_MIA`, etc.).
- **`training`**: target/shadow training loops and defended training variants (DP / adversarial branches, utilities).
- **`model`**: model definitions and wrappers used by training and attacks.
- **`quantization`**: PTQ/QAT quantization pipelines and helpers (including AIMET AdaRound integration).
- **`data`**: dataset download helpers, preparation scripts, and PyTorch `Dataset` wrappers.
- **`dp`**: differential privacy utilities used by DP training paths.
- **`example`**: shell scripts for end-to-end experiments (MIA data collection + attack-only evaluation / defenses).
- **`experiment`**: mechanistic studies behind the paper's analysis — LOOD-style memorization scores (`memorization.py`) and the gradient-norm quantization tables (`grad_norm_quantization_table.py`, `grad_norm_range_normalized_table.py`).
- **`plot`**: figure and table generation for the paper; output PDFs and CSVs land in `plot/figures/`.
- **`main.py`**: unified CLI entry (`--training` switches between training and attack-only modes).
- **`parameter_parser.py`**: all command-line arguments and defaults.
- **`extract_last_layer_grad_norm.py`**: per-sample L2 norms of the last-layer classifier gradient for member and non-member subsets, reused by the `experiment/` tables.

## Demo Experiments

### Membership inference data collection + attack-only MIA (CIFAR-100)

The following pattern matches the repository's intended workflow: **train with `MIA` to populate `save/MIA/...`**, then run `MIA` with `--training None`.

```bash
# Stage 1: train target + shadow and save MIA artifacts
python main.py \
  --dataset_name cifar100 \
  --net_name resnet18 \
  --quantization hawq --a_bit 8 --w_bit 8 \
  --attack_method MIA \
  --training both \
  --num_epochs 50 --trials 3 \
  --device cuda:0

# Stage 2: attack-only MIA evaluation (confidence threshold attack only)
python main.py \
  --dataset_name cifar100 \
  --net_name resnet18 \
  --quantization hawq --a_bit 8 --w_bit 8 \
  --attack_method MIA \
  --compleaknr_attacks confidence \
  --training None \
  --trials 3 \
  --device cuda:0
```

For each attack run, the console will print progress from the corresponding attack implementation (e.g., trial-wise metrics for `MIA`; attack-only runs save to `result/MIA/...`).

#### Choosing which black-box attacks to run

`--compleaknr_attacks` controls the attack-only `MIA` entry:

- **`confidence`** — runs only the confidence threshold attack. It consumes just the saved posteriors, so it is fast and needs no model files.
- **`all`** (the argument's default) — runs all six black-box variants: confidence threshold, entropy threshold, modified entropy, two train-based attacks (LR/RF on posteriors, and on posteriors plus label), and SAMIA.

Note that the two modes have different requirements. The first five variants read only the saved posteriors, but **SAMIA additionally loads the saved target and shadow models** from `save/MIA/.../target/<trial>/model.pth` and `save/MIA/.../shadow/0/model.pth` and queries them on perturbed inputs to estimate output sensitivity. This is still black-box in the sense that only model outputs are used — no gradients or internal activations — but it does require those checkpoints to be present, and it is considerably slower.

If you omit `--compleaknr_attacks`, all six run. Pass `confidence` explicitly, as above, when you only want the fast path or when the model checkpoints are unavailable.

### Batch scripts (defense + MIA)

You can also reproduce larger sweeps via:

```bash
bash example/MIA_cifar100_train.sh
bash example/MIA_cinic10_train.sh
bash example/defense/run_adv.sh
bash example/defense/run_dp.sh
bash example/defense/run_dropout.sh
bash example/RMIA.sh
```

Edit device IDs (`cuda:*`) and hyper-parameters inside those scripts for your cluster.

### Defense examples (CIFAR-100)

Below are minimal examples for defended models, using `--compleaknr_attacks confidence` in attack-only mode.

```bash
# Dropout defense (resnet18_drop010): train target, then black-box confidence attack
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization Original --a_bit 32 --w_bit 32 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

# DP defense (resnet18_dp, eps=5.0): train target, then black-box confidence attack
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 5.0 --trials 1 --device cuda:0

# Adversarial defense (resnet18_adversarial): train target, then black-box confidence attack
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization Original --a_bit 32 --w_bit 32 --num_epochs 50 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
```

For full sweeps over quantization methods and bit-widths, use:
`example/defense/run_dropout.sh`, `example/defense/run_dp.sh`, and `example/defense/run_adv.sh`.

### White-box attacks (two modes)

This repo provides two white-box attack entries:

- `WhiteBoxMIA`: shadow-based white-box MIA (train attack model on shadow features, evaluate on target features).
- `WhiteBoxMIA_Partial`: target-only split mode (no shadow model required; split target features for train/test).

Both load the saved models from `save/MIA/...`, so run them after stage 1 has completed:

```bash
# WhiteBoxMIA (shadow mode)
python main.py \
  --dataset_name cifar100 \
  --net_name resnet18 \
  --quantization hawq --a_bit 8 --w_bit 8 \
  --attack_method WhiteBoxMIA \
  --training None \
  --trials 3 \
  --device cuda:0

# WhiteBoxMIA_Partial (target-only split mode)
python main.py \
  --dataset_name cifar100 \
  --net_name resnet18 \
  --quantization hawq --a_bit 8 --w_bit 8 \
  --attack_method WhiteBoxMIA_Partial \
  --training None \
  --trials 3 \
  --device cuda:0
```

Results are written to `result/WhiteBoxMIA/...` and `result/WhiteBoxMIA_Partial/...`.

## Notes on defaults

The commands above spell out most arguments rather than relying on defaults. A few worth knowing when adapting them (see `parameter_parser.py` for the full list):

| Argument | Default | Note |
| --- | --- | --- |
| `--trials` | `5` | the examples above use `3` or `1` to keep runtimes short |
| `--num_epochs` | `50` | |
| `--batch_size` | `256` | |
| `--optim` / `--lr` | `Adam` / `0.001` | |
| `--random` | `41` | seed for the target/shadow data split; must match between stage 1 and stage 2 |
| `--compleaknr_attacks` | `all` | see the section above |
| `--dataset_name` | `cifar100` | `--dataset` is an accepted alias |
