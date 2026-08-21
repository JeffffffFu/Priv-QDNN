#!/bin/bash

# MIA training for CINIC-10 (target + shadow)
# usage:
#   bash example/MIA_cinic10_train.sh
#
# note:

###############################################################################
# Target training (trials=3)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --attack_method MIA --training target --num_epochs 50 --trials 3 --device cuda:0

###############################################################################
# Shadow training (trials=1)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:2

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --attack_method MIA --training shadow --num_epochs 50 --trials 1 --device cuda:0

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --attack_method MIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization PYT_PTQ --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4

# adaround
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization adaround --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:1
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:1
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:1
python main.py --dataset_name cinic10 --net_name resnet18 --quantization qdrop --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:1

# hawq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4
python main.py --dataset_name cinic10 --net_name resnet18 --quantization hawq --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:4

# pact
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
python main.py --dataset_name cinic10 --net_name resnet18 --quantization pact --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# bsq
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 4 --w_bit 4 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 6 --w_bit 6 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:2
python main.py --dataset_name cinic10 --net_name resnet18 --quantization bsq --a_bit 10 --w_bit 10 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:2





