#!/usr/bin/env bash

# DP defense experiment runner (2-bit only, no Original).
# Dataset: cifar100
# Model:   resnet18_dp
# Eps:     5.0, 1.0
# Epochs:  20
# Batch:   512
# Trials:  1
# Device:  cuda:1

###############################################################################
# Step 1) MIA target training @ 2/2
###############################################################################

# eps = 5.0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1

###############################################################################
# Step 2) MIA black-box (confidence only, attack only) @ 2/2
###############################################################################

# eps = 5.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1

###############################################################################
# Step 3) WhiteBoxMIA (attack only) @ 2/2
###############################################################################

# eps = 5.0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --training None --eps 5.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 2 --w_bit 2 --training None --eps 1.0 --trials 1 --device cuda:1
