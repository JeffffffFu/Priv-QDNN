#!/usr/bin/env bash

# DP defense for Location + mlp_5layer (4-bit).
# Model:   mlp_5layer_dp
# Methods: PYT_PTQ (PQ), AdaRound, HAWQ
# Shadow:  reuse save/MIA/mlp_5layer/location/.../shadow (run MIA_location_4bit shadow first)
# Eps:     5.0, 3.0, 1.0
# Epochs:  20
# Batch:   256
# Trials:  1
# Device:  cuda:3

###############################################################################
# Step 1) MIA target training @ 4/4
###############################################################################

# eps = 5.0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 5.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 5.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 5.0 --trials 1 --device cuda:3

# eps = 3.0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 3.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 3.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 3.0 --trials 1 --device cuda:3

# eps = 1.0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 1.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 1.0 --trials 1 --device cuda:3
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 20 --batch_size 256 --eps 1.0 --trials 1 --device cuda:3

###############################################################################
# Step 2) MIA black-box (confidence only, attack only) @ 4/4
# Shadow cache: save/MIA/mlp_5layer/location/.../shadow
###############################################################################

# eps = 5.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3

# eps = 3.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3

# eps = 1.0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3

###############################################################################
# Step 3) WhiteBoxMIA (attack only) @ 4/4
###############################################################################

# eps = 5.0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 5.0 --trials 1 --device cuda:3

# eps = 3.0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 3.0 --trials 1 --device cuda:3

# eps = 1.0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization adaround --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization hawq --a_bit 4 --w_bit 4 --training None --eps 1.0 --trials 1 --device cuda:3
