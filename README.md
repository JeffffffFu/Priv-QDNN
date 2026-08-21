# Priv-QDNN

This repository contains a research codebase for **privacy evaluation of quantized deep neural networks**. It supports training target/shadow models under common post-training quantization (PTQ) and quantization-aware training (QAT) settings, saving intermediate attack features to disk, and running a suite of **membership-inference-related** attacks and metrics (e.g., `MIA`, `RMIA`, `WhiteBoxMIA`, `WhiteBoxMIA_Partial`). The attack-only `MIA` pipeline uses black-box inference-time outputs (e.g., saved posteriors / confidence-style signals), not white-box gradients or full internal activations of the target model. The codebase also integrates **Opacus**-based differential privacy training paths for defended models (e.g., dropout / DP / adversarial variants in the provided example scripts).

The evaluation workflow is split into two stages: **(1) training** (`--training target|shadow|both`) to materialize `save/<attack_method>/...` artifacts, and **(2) attack-only evaluation** (`--training None`) that loads saved posteriors/models and reports metrics under `result/MIA/...`.

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

## Code Structure of Priv-QDNN

- `**attack`**: attack implementations and evaluation entry points (`MIA`, `RMIA`, `WhiteBox_MIA`, etc.).
- `**training`**: target/shadow training loops and defended training variants (DP / adversarial branches, utilities).
- `**model`**: model definitions and wrappers used by training and attacks.
- `**quantization**`: PTQ/QAT quantization pipelines and helpers (including AIMET AdaRound integration).
- `**data**`: dataset download helpers and PyTorch `Dataset` wrappers.
- `**dp**`: differential privacy utilities used by DP training paths.
- `**example**`: shell scripts for end-to-end experiments (MIA data collection + attack-only evaluation / defenses).
- `**main.py**`: unified CLI entry (`--training` switches between training and attack-only modes).
- `**parameter_parser.py**`: all command-line arguments and defaults.

## Demo Experiments

### Membership inference data collection + attack-only MIA (CIFAR-100)

The attack-only `MIA` entry in this repo runs a **black-box membership inference pipeline**: stage 2 below only consumes artifacts such as saved posteriors / confidence-based features—no white-box access to the target network during the attack.

The following pattern matches the repository’s intended workflow: **train with `MIA` to populate `save/MIA/...`**, then run `MIA` with `--training None`.

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

# Stage 2: attack-only MIA evaluation
python main.py \
  --dataset_name cifar100 \
  --net_name resnet18 \
  --quantization hawq --a_bit 8 --w_bit 8 \
  --attack_method MIA \
  --training None \
  --trials 3 \
  --device cuda:0
```

For each attack run, the console will print progress from the corresponding attack implementation (e.g., trial-wise metrics for `MIA`; attack-only runs save to `result/MIA/...`).

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

Below are minimal examples for defended models. For black-box MIA, this repo uses
`--compleaknr_attacks confidence` in attack-only mode.

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

You can run them after stage-1 artifacts are ready under `save/MIA/...`:

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