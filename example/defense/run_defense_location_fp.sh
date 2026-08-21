#!/usr/bin/env bash

# Location + mlp_5layer defense experiments (Full-precision / Original only).
# Quantization: Original_32_32  (--quantization Original --a_bit 32 --w_bit 32)
# Defenses:     Adversarial, Dropout 10%, Dropout 15%, DP (eps=1, 5)
# Attacks:      MIA (confidence), WhiteBoxMIA
# Shadow:       reuse save/MIA/mlp_5layer/location/Original_32_32/shadow if available
# Epochs:        20
# Trials:        1
# Device:        cuda:0  (DP uses batch_size 256)

set -euo pipefail

cd "$(dirname "$0")/../.."

###############################################################################
# Adversarial (mlp_5layer_adversarial)
###############################################################################

# Optional: pretrain CT attack model once (uses mlp_5layer FP shadow posteriors)
# python -m training.pretrain_mia_attack_model --dataset_name location --shadow_net_name mlp_5layer --quantization Original --a_bit 32 --w_bit 32 --num_epochs 100 --device cuda:0

# 1) Target training
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_adversarial --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --trials 1 --device cuda:0

# 2) MIA black-box (confidence)
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_adversarial --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

# 3) WhiteBoxMIA
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_adversarial --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

###############################################################################
# Dropout 10% (mlp_5layer_drop010)
###############################################################################

python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop010 --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop010 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop010 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

###############################################################################
# Dropout 15% (mlp_5layer_drop015)
###############################################################################

python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_drop015 --quantization Original --a_bit 32 --w_bit 32 --training None --trials 1 --device cuda:0

###############################################################################
# DP (mlp_5layer_dp) — eps = 5.0, 1.0
###############################################################################

# eps = 5.0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 256 --eps 5.0 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 5.0 --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 5.0 --trials 1 --device cuda:0

# eps = 1.0
python main.py --training target --attack_method MIA --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --num_epochs 20 --batch_size 256 --eps 1.0 --trials 1 --device cuda:0
python main.py --attack_method MIA --compleaknr_attacks confidence --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 1.0 --trials 1 --device cuda:0
python main.py --attack_method WhiteBoxMIA --dataset_name location --net_name mlp_5layer_dp --quantization Original --a_bit 32 --w_bit 32 --training None --eps 1.0 --trials 1 --device cuda:0
