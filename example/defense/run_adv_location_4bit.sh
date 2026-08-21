#!/usr/bin/env bash

# Adversarial defense for Location + mlp_5layer (4-bit).
# Model:   mlp_5layer_adversarial
# Methods: PYT_PTQ (PQ), AdaRound, HAWQ
# Shadow:  reuse save/MIA/mlp_5layer/location/.../shadow (run MIA_location_4bit shadow first)
# Epochs:  20
# Trials:  1
# Device:  cuda:0

###############################################################################
# Step 0) Pretrain CT attack model (once, before adversarial target training)
# Uses shadow posteriors from mlp_5layer @ 4/4.
###############################################################################

#python -m training.pretrain_mia_attack_model --dataset_name location --shadow_net_name mlp_5layer --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 100 --device cuda:0

###############################################################################
# Step 1) MIA target training @ 4/4
###############################################################################

python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_adversarial --quantization adaround --a_bit 4 --w_bit 4 --num_epochs 20 --trials 1 --device cuda:0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_adversarial --quantization hawq --a_bit 4 --w_bit 4 --num_epochs 10 --trials 1 --device cuda:0

###############################################################################
# Step 2) MIA black-box (confidence only, attack only) @ 4/4
# Shadow cache: save/MIA/mlp_5layer/location/.../shadow
###############################################################################

python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_adversarial --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_adversarial --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0

###############################################################################
# Step 3) WhiteBoxMIA (attack only) @ 4/4
###############################################################################

python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_adversarial --quantization PYT_PTQ --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_adversarial --quantization adaround --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_adversarial --quantization hawq --a_bit 4 --w_bit 4 --training None --trials 1 --device cuda:0
