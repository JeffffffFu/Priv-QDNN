#!/usr/bin/env bash

# DP defense experiment runner (explicit commands, no loop/variable replacement).
# Dataset: cifar100
# Model:   resnet18_dp
# Epochs:  20
# Batch:   512
# Trials:  1
# Device:  cuda:1

###############################################################################
# Step 1) MIA target training
###############################################################################

# eps = 5.0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 5.0 --trials 1 --device cuda:1

# eps = 3.0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 3.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --num_epochs 20 --batch_size 512 --eps 1.0 --trials 1 --device cuda:1

###############################################################################
# Step 2) MIA black-box (confidence only, attack only)
###############################################################################

# eps = 5.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1

# eps = 3.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1
###############################################################################
# Step 3) WhiteBoxMIA (attack only)
###############################################################################

# eps = 5.0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 5.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 5.0 --trials 1 --device cuda:1

# eps = 3.0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 3.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 3.0 --trials 1 --device cuda:1

# eps = 1.0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization adaround --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 6 --w_bit 6 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 8 --w_bit 8 --training None --eps 1.0 --trials 1 --device cuda:1
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_dp --quantization hawq --a_bit 10 --w_bit 10 --training None --eps 1.0 --trials 1 --device cuda:1

