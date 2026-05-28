model_name=Union
exp_name=Union_pretrain_x128
wandb_mode=disabled
d_model=128
random_port=$((RANDOM % 9000 + 1000))
ckpt_path=checkpoints/pretrain_ckpt.pth
offset=384

torchrun --nnodes 1 --nproc-per-node=1 --master_port $random_port run.py \
  --is_training 0 \
  --zero_shot_forecasting_new_length unify \
  --max_offset 384 \
  --offset $offset \
  --model_id $exp_name \
  --model $model_name \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --debug $wandb_mode \
  --pretrained_weight $ckpt_path \
  --task_data_config_path data_provider/multitask_zero_shot_new_length.yaml
