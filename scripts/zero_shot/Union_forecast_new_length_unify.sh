#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 PATH_TO_CHECKPOINT [additional run.py options]" >&2
  echo "Compatibility entry: evaluates a compatible checkpoint; no zero-shot adaptation." >&2
  exit 2
fi
checkpoint=$1
shift
echo "Running checkpoint evaluation. No zero-shot adaptation protocol is applied."

exec python run.py \
  --is_training 0 --model_id Union_checkpoint_eval --model Union \
  --pretrained_weight "$checkpoint" \
  --task_data_config_path data_provider/multitask_zero_shot_new_length.yaml \
  --fix_seed 2021 --batch_size 32 \
  --d_model 128 --e_layers 3 --n_heads 4 \
  --patch_len 16 --stride 16 --prompt_num 10 \
  "$@"
