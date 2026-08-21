#!/bin/bash

# MIA training + attack for Texas100 (2-bit only)
# usage:
#   bash example/MIA_texas_2bit.sh
#
# note:
#   - model: mlp_5layer
#   - quantization: 6 methods @ 2/2 (Original already done)
#   - prepare data first: python data/prepare_tabular.py --dataset texas
#   - device: cuda:2
#   - num_epochs: 50
#   - trials: 1 (target / shadow / attack)

###############################################################################
# Target training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

# adaround
python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

# hawq
python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

# pact
python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

###############################################################################
# Shadow training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# adaround
python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# hawq
python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# pact
python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

# PYT_PTQ
python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

# adaround
python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

# hawq
python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

# pact
python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:2

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

# adaround
python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

# hawq
python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

# pact
python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:2

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2

# adaround
python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2

# hawq
python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2

# pact
python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:2
