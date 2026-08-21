#!/bin/bash

# RMIA for CINIC-10 (4-bit only)
# usage:
#   bash example/RMIA_cinic10_4bit.sh
#
# note:
#   - model: mobilenetv2
#   RMIA don't need --training, it will train when it attack
#   observations=64, num_epochs=60, trials=1


# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 60 --observations 64 --trials 1 --device cuda:5
