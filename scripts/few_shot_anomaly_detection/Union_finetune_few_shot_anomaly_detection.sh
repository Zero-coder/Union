model_name=Union
wandb_mode=disabled
project_name=fewshot_anomaly_detection
exp_name=fewshot_anomaly_finetune
random_port=$((RANDOM % 9000 + 1000))

torchrun --nnodes 1 --nproc-per-node=1 --master_port $random_port run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --train_epochs 5 \
  --learning_rate 1e-4 \
  --batch_size 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --task_data_config_path data_provider/anomaly_detection.yaml
