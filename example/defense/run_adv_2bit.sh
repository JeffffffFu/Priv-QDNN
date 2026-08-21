#!/usr/bin/env bash

# Adversarial-defense experiments (2-bit only, no Original).
# Dataset: cifar100
# Model:   resnet18_adversarial
# Quant:   PYT_PTQ, adaround, hawq @ 2/2
# Epochs:  50
# Trials:  1
# Device:  cuda:3

###############################################################################
# Step 1) MIA target training @ 2/2
###############################################################################

python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 2 --w_bit 2 --num_epochs 50 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 2 --w_bit 2 --num_epochs 50 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 2 --w_bit 2 --num_epochs 50 --trials 1 --device cuda:3

###############################################################################
# Step 2) MIA black-box (confidence only, training None) @ 2/2
###############################################################################

python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3

###############################################################################
# Step 3) WhiteBoxMIA attack (training None) @ 2/2
###############################################################################

python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:3
