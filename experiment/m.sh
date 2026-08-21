#!/usr/bin/env bash

# LOOD memorization score (experiment/memorization.py)
# Dataset: cifar100
# Model:   resnet18 (no defense)
# Bit:     2/2
# Methods: PYT_PTQ, adaround, qdrop, hawq, pact, bsq (6 quant methods)
# Metric:  KL( softmax(f_D(z)) || softmax(f_{D\\z}(z)) ), Q=z
# Output:  result/memorization_lood/cifar100/resnet18/{quant}_2_2/<timestamp>/
#
# Params:  train_pool_size=4000, num_loo=40, num_epochs=50, seed=42
# Cost:    1 full train on D + num_loo leave-one-out retrains (+ quant each time).

cd "$(dirname "$0")/.." || exit 1

###############################################################################
# Step 0) Quick smoke test (~5 LOO x 3 epochs)
###############################################################################

# python experiment/memorization.py --quick --device cuda:0

###############################################################################
# Step 1) resnet18 @ 2/2 — six quantization methods
###############################################################################

python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization PYT_PTQ --a_bit 2 --w_bit 2 --device cuda:0 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization adaround --a_bit 2 --w_bit 2 --device cuda:0 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization qdrop --a_bit 2 --w_bit 2 --device cuda:2 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization hawq --a_bit 2 --w_bit 2 --device cuda:3 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization pact --a_bit 2 --w_bit 2 --device cuda:0 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization bsq --a_bit 2 --w_bit 2 --device cuda:0 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42

###############################################################################
# Step 2) Full-precision reference (optional)
###############################################################################

# python experiment/memorization.py --dataset_name cifar100 --net_name resnet18 --quantization Original --a_bit 32 --w_bit 32 --device cuda:0 --train_pool_size 4000 --num_loo 40 --num_epochs 50 --seed 42
