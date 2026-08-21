#!/bin/bash

# RMIA for Texas100 / Location (4-bit only)
# usage:
#   bash example/RMIA_texas_location_4bit.sh
#
# note:
#   - model: mlp_5layer
#   - quantization: 6 methods @ 4/4 (Original already done)
#   - prepare data first: python data/prepare_tabular.py --dataset both
#   - device: cuda:4
#   - observations=64, num_epochs=50, trials=1
#   - RMIA don't need --training, it will train when it attack

###############################################################################
# Texas100
###############################################################################

# PYT_PTQ
# python main.py --dataset_name texas --net_name mlp_5layer --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

# # adaround
# python main.py --dataset_name texas --net_name mlp_5layer --quantization adaround --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

# # qdrop
# python main.py --dataset_name texas --net_name mlp_5layer --quantization qdrop --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

# # hawq
# python main.py --dataset_name texas --net_name mlp_5layer --quantization hawq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

# # pact
# python main.py --dataset_name texas --net_name mlp_5layer --quantization pact --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

# # bsq
# python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 1 --device cuda:4

###############################################################################
# Location
###############################################################################

# PYT_PTQ
python main.py --dataset_name location --net_name mlp_5layer --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4

# adaround
python main.py --dataset_name location --net_name mlp_5layer --quantization adaround --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4

# qdrop
python main.py --dataset_name location --net_name mlp_5layer --quantization qdrop --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4

# hawq
python main.py --dataset_name location --net_name mlp_5layer --quantization hawq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4

# pact
python main.py --dataset_name location --net_name mlp_5layer --quantization pact --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 20 --observations 64 --trials 1 --device cuda:4
