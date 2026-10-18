#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

# Compatibility filename: runs masked reconstruction pretraining only.
# For subsequent full fine-tuning, pass the saved checkpoint to a few-shot script.
# There is no prompt-only training stage or automatic checkpoint discovery.
exec python run_pretrain.py \
  --is_training 1 --model_id Union_pretrain_x128 --model Union \
  --task_data_config_path data_provider/multi_task_pretrain.yaml \
  --fix_seed 2021 --train_epochs 10 --batch_size 32 --acc_it 1 \
  --learning_rate 5e-5 --min_lr 1e-6 --weight_decay 5e-6 \
  --min_mask_ratio 0.7 --max_mask_ratio 0.8 \
  --d_model 128 --e_layers 3 --n_heads 4 \
  --patch_len 16 --stride 16 --prompt_num 10 \
  "$@"
