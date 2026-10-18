#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Reads the ETTh1 CSV specified in configs/forecast_csv.yaml.
exec python run.py \
  --is_training 1 --model_id Union_ETTh1 --model Union \
  --task_data_config_path configs/forecast_csv.yaml \
  --fix_seed 2021 --train_epochs 10 --batch_size 32 \
  --learning_rate 1e-4 --min_lr 1e-6 --weight_decay 1e-5 \
  --d_model 128 --e_layers 3 --n_heads 4 \
  --patch_len 16 --stride 16 --prompt_num 10 \
  "$@"
