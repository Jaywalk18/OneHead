#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:?usage: bash examples/cifar10_smoke.sh /path/to/data}"
DEVICE="${DEVICE:-cuda:0}"

python src/simclr_cifar.py \
  --dataset cifar10 \
  --data-root "$DATA_ROOT" \
  --head mlp \
  --epochs 1 \
  --batch-size 128 \
  --num-workers 2 \
  --seed 0 \
  --device "$DEVICE" \
  --out-dir outputs/cifar10_smoke
