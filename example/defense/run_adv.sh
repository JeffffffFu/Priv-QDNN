#!/usr/bin/env bash

# Adversarial-defense experiments (explicit commands, no loop replacement).
# Dataset: cifar100
# Model:   resnet18_adversarial
# Epochs:  50
# Trials:  1
# Device:  cuda:0

###############################################################################
# Step 1) MIA target training
###############################################################################

# Baseline (Original)
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization Original --a_bit 32 --w_bit 32 --num_epochs 50 --trials 1 --device cuda:0

# Core methods @ 4/6/8/10-bit
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 6 --w_bit 6 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 10 --w_bit 10 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 6 --w_bit 6 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 10 --w_bit 10 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 6 --w_bit 6 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 10 --w_bit 10 --num_epochs 50 --trials 1 --device cuda:0

# Extra methods @ 8-bit only
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization qdrop --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization pact --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset cifar100 --net_name resnet18_adversarial --quantization bsq --a_bit 8 --w_bit 8 --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# Step 2) MIA black-box (confidence only, training None)
###############################################################################

# Baseline (Original)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

# Core methods @ 4/6/8/10-bit
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0

# Extra methods @ 8-bit only
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization qdrop --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization pact --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset cifar100 --net_name resnet18_adversarial --quantization bsq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
###############################################################################
# Step 3) WhiteBoxMIA attack (training None)
###############################################################################

# Baseline (Original)
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

# Core methods @ 4/6/8/10-bit
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization PYT_PTQ --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization adaround --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 6 --w_bit 6 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 8 --w_bit 8 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name cifar100 --net_name resnet18_adversarial --quantization hawq --a_bit 10 --w_bit 10 --training None --trials 1 --device cuda:0

