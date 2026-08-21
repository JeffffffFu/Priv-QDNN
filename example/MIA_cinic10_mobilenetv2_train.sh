#!/bin/bash

# MIA training for CINIC-10 with MobileNetV2 (target + shadow + attacks)
# usage:
#   bash example/MIA_cinic10_mobilenetv2_train.sh   # all on cuda:0
#   or run in parallel:
#     bash example/run3.sh   # target,  cuda:1
#     bash example/run4.sh   # shadow,  cuda:2
#     bash example/run5.sh   # black-box MIA, cuda:3  (after run3+run4)
#     bash example/run6.sh   # white-box, cuda:4      (after run3+run4)
#
# note:
#   - model: mobilenetv2 (torchvision backbone, 10-class head)
#   - quantization: Original 32/32 + 8-bit PTQ/QAT (no bsq)
#   - hawq uses q_simple_cnn wrapper (classifier-focused QAT)

###############################################################################
# Target training (trials=3)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training target --num_epochs 60 --trials 3 --device cuda:0

###############################################################################
# Shadow training (trials=1)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training shadow --num_epochs 60 --trials 1 --device cuda:0

###############################################################################
# Black-box attack (MIA entry; CompLeakNR pipeline)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method MIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:0

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

# Original
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# PYT_PTQ
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization PYT_PTQ --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# adaround
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization adaround --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# qdrop
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization qdrop --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# hawq
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization hawq --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0

# pact
python main.py --dataset_name cinic10 --net_name mobilenetv2 --quantization pact --a_bit 8 --w_bit 8 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:0
