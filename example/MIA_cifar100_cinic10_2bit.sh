#!/bin/bash

# MIA training for CIFAR-100 & CINIC-10 (2-bit only, no attack)
# usage:
#   bash example/MIA_cifar100_cinic10_2bit.sh
#
# note:
#   - model: resnet18
#   - quantization: 6 methods @ 2/2 (Original already done)
#   - cifar100 num_epochs: 50
#   - cinic10 num_epochs: 50
#   - trials: 1

###############################################################################
# CIFAR-100 — Target training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# CIFAR-100 — Shadow training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# pact
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# CINIC-10 — Target training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1

###############################################################################
# CINIC-10 — Shadow training (trials=1)
###############################################################################

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1
