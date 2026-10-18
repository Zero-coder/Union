#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 PATH_TO_SOURCE_CHECKPOINT [additional run.py options]" >&2
  exit 2
fi
checkpoint=$1
shift

# Requires the documented anomaly NPZ. Thresholds use validation scores.
exec python run.py \
  --is_training 1 --model_id Union_anomaly_fewshot --model Union \
  --pretrained_weight "$checkpoint" --subsample_pct 0.20 --anomaly_ratio 1.0 \
  --task_data_config_path data_provider/anomaly_detection.yaml \
  --fix_seed 2021 --train_epochs 5 --batch_size 32 \
  --learning_rate 1e-4 --min_lr 1e-6 --weight_decay 1e-5 \
  --d_model 128 --e_layers 3 --n_heads 4 \
  --patch_len 16 --stride 16 --prompt_num 10 \
  "$@"
