#!/usr/bin/env bash

# Dropout defense for Location + mlp_5layer (4-bit).
# Models:  mlp_5layer_drop015 / mlp_5layer_drop020
# Methods: Original (FP), PYT_PTQ (PQ), AdaRound, HAWQ
# Shadow:  reuse save/MIA/mlp_5layer/location/.../shadow (run MIA_location_4bit shadow first)
# Epochs:  20
# Trials:  1
# Device:  cuda:0

###############################################################################
# mlp_5layer_drop015 @ 4/4
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop015 --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop015 --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0

# 2) MIA black-box (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop015 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop015 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0

# 3) WhiteBoxMIA
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop015 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop015 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop015 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0

###############################################################################
# mlp_5layer_drop020 @ 4/4
###############################################################################

# 1) MIA target training
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop020 --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop020 --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop020 --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0

# 2) MIA black-box (confidence only)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop020 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop020 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop020 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0

# 3) WhiteBoxMIA
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop020 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop020 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop020 --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop020 --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
