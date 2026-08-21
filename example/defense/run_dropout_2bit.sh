#!/usr/bin/env bash

# Dropout defense experiments (2-bit only, no Original).
# Dataset: cifar100
# Models:  resnet18_drop010 / resnet18_drop015
# Trials:  1
# Device:  cuda:2

###############################################################################
# resnet18_drop010 @ 2/2
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:2 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:2 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:2 --trials 1

# 2) Black-box: MIA (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2

# 3) White-box attacks
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2

###############################################################################
# resnet18_drop015 @ 2/2
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:2 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:2 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:2 --trials 1

# 2) Black-box: MIA (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2

# 3) White-box attacks
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:2
