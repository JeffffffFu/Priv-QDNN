#!/bin/bash

# RMIA for CIFAR-100 (4-bit only)
# usage:
#   bash example/RMIA_cifar100_4bit.sh
#
# note:
#   RMIA don't need --training, it will train when it attack
#   observations=64, num_epochs=50, trials=3


# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method RMIA --num_epochs 50 --observations 64 --trials 3 --device cuda:0
