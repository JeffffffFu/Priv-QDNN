#!/usr/bin/env bash

# MIA training + attack for CINIC-10 + MobileNetV2 (8/6/4/2-bit)
# usage:
#   bash example/MIA_cinic10_mobilenetv2_2468bit.sh
#
# note:
#   - model: mobilenetv2
#   - quantization: 6 methods @ 8/6/4/2 (Original FP: see MIA_cinic10_mobilenetv2_train.sh)
#   - num_epochs: 50
#   - trials: target=1, shadow=1, attack=1
#   - device: cuda:0

set -euo pipefail

cd "$(dirname "$0")/.."

###############################################################################
# 8-bit | Target training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 8-bit | Shadow training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 8-bit | Attack-only evaluation (MIA)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# 8-bit | WhiteBoxMIA attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# 8-bit | WhiteBoxMIA_Partial attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

###############################################################################
# 6-bit | Target training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 6-bit | Shadow training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 6-bit | Attack-only evaluation (MIA)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# 6-bit | WhiteBoxMIA attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# 6-bit | WhiteBoxMIA_Partial attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

###############################################################################
# 4-bit | Target training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 4-bit | Shadow training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 4-bit | Attack-only evaluation (MIA)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# 4-bit | WhiteBoxMIA attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# 4-bit | WhiteBoxMIA_Partial attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0

###############################################################################
# 2-bit | Target training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training target --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 2-bit | Shadow training (trials=1, num_epochs=50)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# 2-bit | Attack-only evaluation (MIA)
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 2 --w_bit 2 --attack_method MIA --training None --trials 1 --device cuda:0

###############################################################################
# 2-bit | WhiteBoxMIA attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA --training None --trials 1 --device cuda:0

###############################################################################
# 2-bit | WhiteBoxMIA_Partial attack
###############################################################################

python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization bsq --a_bit 2 --w_bit 2 --attack_method WhiteBoxMIA_Partial --training None --trials 1 --device cuda:0
