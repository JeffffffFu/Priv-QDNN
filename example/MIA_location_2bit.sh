#!/bin/bash

# MIA training + attack for Location (2-bit only)
# usage:
#   bash example/MIA_location_2bit.sh
#
# note:
#   - model: mlp_5layer
#   - quantization: 6 methods @ 2/2 (Original already done)
#   - prepare data first: python data/prepare_tabular.py --dataset location
#   - device: cuda:3
#   - num_epochs: 20 (target/shadow); hawq/bsq target use 10 epochs

###############################################################################
# Target training (trials=3)
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:3

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:3

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:3

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 10 --trials 3 --device cuda:3

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:3

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 10 --trials 3 --device cuda:3

###############################################################################
# Shadow training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 10 --trials 1 --device cuda:3

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 3 --device cuda:3

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3
