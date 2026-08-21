#!/usr/bin/env bash
# Dropout defense experiments (drop010 / drop015 / drop020).
# Quantization bits: 2/4/6/8 (+ Original 32/32 for target training).
# Dataset: cifar100, model: resnet18_dropXXX

###############################################################################
# resnet18_drop010
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization Original --a_bit 32 --w_bit 32 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 8 --w_bit 8 --device cuda:1 --trials 1

# 2) Black-box: MIA (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1

# 3) White-box attacks
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop010 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1

###############################################################################
# resnet18_drop015
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization Original --a_bit 32 --w_bit 32 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 8 --w_bit 8 --device cuda:1 --trials 1

# 2) Black-box: MIA (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
# 3) White-box attacks
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop015 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1

###############################################################################
# resnet18_drop020
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization Original --a_bit 32 --w_bit 32 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 4 --w_bit 4 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 6 --w_bit 6 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 8 --w_bit 8 --device cuda:1 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:3 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 4 --w_bit 4 --device cuda:3 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 6 --w_bit 6 --device cuda:3 --trials 1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 8 --w_bit 8 --device cuda:3 --trials 1

# 2) Black-box: MIA (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1

# 3) White-box attacks
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 2 --w_bit 2 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_drop020 --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:1

