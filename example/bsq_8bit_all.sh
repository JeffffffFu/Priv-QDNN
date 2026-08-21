#!/bin/bash

# BSQ 8-bit full experiments
# usage:
#   bash example/bsq_8bit_all.sh
#
# note:
#   - CINIC-10 + mobilenetv2  (same as MIA_cinic10_mobilenetv2_train.sh)
#   - Texas100 / Location + mlp_5layer  (same as MIA_texas_location_train.sh)
#   - quantization: bsq 8/8 only
#   - num_epochs: 50
#   - prepare tabular data first: python data/prepare_tabular.py --dataset both

###############################################################################
# CINIC-10 + mobilenetv2 - Target training (trials=3)
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

###############################################################################
# CINIC-10 + mobilenetv2 - Shadow training (trials=1)
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# CINIC-10 + mobilenetv2 - Black-box attack (MIA entry; CompLeakNR pipeline)
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

###############################################################################
# CINIC-10 + mobilenetv2 - WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

###############################################################################
# CINIC-10 + mobilenetv2 - WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

###############################################################################
# CINIC-10 + mobilenetv2 - RMIA attack
# RMIA does not use --training; shadow models are trained inside the attack.
###############################################################################

# bsq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method RMIA --num_epochs 50 --trials 3 --observations 64 --device cuda:0

###############################################################################
# Texas100 + mlp_5layer - Target training (trials=3)
###############################################################################

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

###############################################################################
# Texas100 + mlp_5layer - Shadow training (trials=1)
###############################################################################

# bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# Location + mlp_5layer - Target training (trials=3)
###############################################################################

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:0

###############################################################################
# Location + mlp_5layer - Shadow training (trials=1)
###############################################################################

# bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:0

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

# Texas100 - bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

# Location - bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# Texas100 - bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# Location - bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# Texas100 - bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# Location - bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

###############################################################################
# RMIA attack (train shadow + target + attack in one run)
# RMIA does not use --training; shadow models are trained inside the attack.
###############################################################################

# Texas100 - bsq
python main.py --dataset_name texas --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method RMIA --num_epochs 50 --trials 3 --observations 64 --device cuda:0

# Location - bsq
python main.py --dataset_name location --net_name mlp_5layer --quantization bsq --a_bit 8 --w_bit 8 --attack_method RMIA --num_epochs 20 --trials 3 --observations 64 --device cuda:0
