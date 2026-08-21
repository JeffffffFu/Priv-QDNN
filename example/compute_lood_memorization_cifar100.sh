#!/usr/bin/env bash

# LOOD memorization @ 2-bit — resnet18 only (uses experiment/memorization.py)
# Wrapper: same commands as experiment/m.sh

cd "$(dirname "$0")/.." || exit 1
bash experiment/m.sh
