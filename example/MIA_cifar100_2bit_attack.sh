#!/bin/bash

# MIA attack-only for CIFAR-100 (2-bit only)
# usage:
#   bash example/MIA_cifar100_2bit_attack.sh
#
# note:
#   - model: resnet18
#   - quantization: 6 methods @ 2/2
#   - trials: 1
#   - requires target + shadow models already trained

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
