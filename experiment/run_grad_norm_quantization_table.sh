#!/usr/bin/env bash
# Gradient-norm quantization table (Table 5 style): run from the Priv-QDNN root.
# One command per line; comment out or copy them as needed.
# Requires an existing model.pth or target_model.pkl under
# save/MIA/<net>/<dataset>/<quant>_<a>_<w>/target/
#
# Output (default layout, kept separate so runs do not overwrite each other):
#   - FP cache:    result/grad_norm_quant_table/<dataset>/<net>/_fp_cache/r<random>_t<trial>/*.npy
#   - single cell: result/grad_norm_quant_table/<dataset>/<net>/<quant>_<a>_<w>/single_cell.json
#   - full table:  result/grad_norm_quant_table/<dataset>/<net>/run_<timestamp>/grad_norm_diff_table.csv

cd "$(dirname "$0")/.." || exit 1

# -----------------------------------------------------------------------------
# 1) Run the whole table at once (joint FP min-max, then the non-mem mean difference
#    per quantized cell; missing models become nan)
# -----------------------------------------------------------------------------
python experiment/grad_norm_quantization_table.py --full_table --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

# -----------------------------------------------------------------------------
# 2) Without joint FP-Q normalization (raw norm differences, for comparison)
# -----------------------------------------------------------------------------
# python experiment/grad_norm_quantization_table.py --full_table --no_normalize --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

# -----------------------------------------------------------------------------
# 3) Single-cell smoke run (example: AdaRound 8-bit)
# -----------------------------------------------------------------------------
# python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

# -----------------------------------------------------------------------------
# 4) Cell by cell, following the paper's table (equivalent to splitting --full_table;
#    uncomment only if you want to run it in batches)
# -----------------------------------------------------------------------------
python experiment/grad_norm_quantization_table.py --quantization PYT_PTQ --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization PYT_PTQ --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization PYT_PTQ --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization PYT_PTQ --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_quantization_table.py --quantization bsq --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization bsq --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization bsq --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization bsq --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_quantization_table.py --quantization pact --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization pact --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization pact --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization pact --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_quantization_table.py --quantization qdrop --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization qdrop --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization qdrop --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization qdrop --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_quantization_table.py --quantization hawq --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization hawq --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization hawq --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization hawq --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --device cuda:0

# -----------------------------------------------------------------------------
# 5) New metric: score = (mean(non)-mean(mem)) / (max(mem∪non)-min(mem∪non))
# -----------------------------------------------------------------------------
# Path check (verified):
#   under save/MIA/resnet18/cifar100/ there are
#   Original_32_32 and the quantization directories (PYT_PTQ/adaround/bsq/pact/qdrop/hawq, bits=4/6/8/10),
#   with target/0/model.pth available.
#
# 5.1 Full table (sweeps full precision + every quantization; writes to result/grad_norm_range_norm_table/...)
python experiment/grad_norm_range_normalized_table.py --full_table --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

# 5.2 Cell by cell (full precision + every quantization)
python experiment/grad_norm_range_normalized_table.py --quantization Original --a_bit 32 --w_bit 32 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization PYT_PTQ --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization PYT_PTQ --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization PYT_PTQ --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization PYT_PTQ --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization adaround --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization adaround --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization adaround --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization adaround --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization bsq --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization bsq --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization bsq --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization bsq --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization pact --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization pact --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization pact --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization pact --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization qdrop --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization qdrop --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization qdrop --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization qdrop --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization hawq --a_bit 4 --w_bit 4 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization hawq --a_bit 6 --w_bit 6 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization hawq --a_bit 8 --w_bit 8 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization hawq --a_bit 10 --w_bit 10 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0

# -----------------------------------------------------------------------------
# 6) 2-bit, cell by cell (requires existing models under save/MIA/resnet18/cifar100/*_2_2/target/)
# -----------------------------------------------------------------------------
python experiment/grad_norm_quantization_table.py --quantization PYT_PTQ --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization adaround --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization bsq --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization pact --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization qdrop --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0
python experiment/grad_norm_quantization_table.py --quantization hawq --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --device cuda:0

python experiment/grad_norm_range_normalized_table.py --quantization PYT_PTQ --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization adaround --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization bsq --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization pact --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization qdrop --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
python experiment/grad_norm_range_normalized_table.py --quantization hawq --a_bit 2 --w_bit 2 --dataset_name cifar100 --net_name resnet18 --attack_method MIA --random 42 --trial 0 --device cuda:0
