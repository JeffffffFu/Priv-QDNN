#!/usr/bin/env bash

# MIA training + attack for CIFAR-100 + ResNet18 (4-bit only)
# usage:
#   bash example/MIA_cifar100_4bit.sh
#
# note:
#   - model: resnet18
#   - quantization: 6 methods @ 4/4 (Original already done)
#   - num_epochs: 50
#   - trials: target=1, shadow=1, attack=1

set -euo pipefail

cd "$(dirname "$0")/.."

###############################################################################
# Target training (trials=1)
###############################################################################

python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:1
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:2

###############################################################################
# Shadow training (trials=1)
###############################################################################

python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:1
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

python main.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
