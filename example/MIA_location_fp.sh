#!/usr/bin/env bash

# MIA training + attack for Location + mlp_5layer (Full-precision / Original only)
# usage:
#   bash example/MIA_location_fp.sh
#
# note:
#   - model: mlp_5layer
#   - quantization: Original_32_32  (--quantization Original --a_bit 32 --w_bit 32)
#   - prepare data first: python data/prepare_tabular.py --dataset location
#   - num_epochs: 20
#   - trials: target=3, shadow=1, attack=3
#   - device: cuda:3

set -euo pipefail

cd "$(dirname "$0")/.."

###############################################################################
# Target training (trials=3)
###############################################################################

python main.py --dataset_name location --net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training target --num_epochs 20 --trials 3 --device cuda:3

###############################################################################
# Shadow training (trials=1)
###############################################################################

python main.py --dataset_name location --net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training shadow --num_epochs 20 --trials 1 --device cuda:3

###############################################################################
# Attack-only evaluation (MIA entry; compatible with CompLeakNR pipeline)
###############################################################################

python main.py --dataset_name location --net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --attack_method MIA --training None --trials 3 --device cuda:3

###############################################################################
# WhiteBoxMIA attack (shadow mode, last-layer grad)
###############################################################################

python main.py --dataset_name location --net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA --training None --trials 3 --device cuda:3

###############################################################################
# WhiteBoxMIA_Partial attack (no-shadow, last-layer grad)
###############################################################################

python main.py --dataset_name location --net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --attack_method WhiteBoxMIA_Partial --training None --trials 3 --device cuda:3
