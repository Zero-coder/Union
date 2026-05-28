import argparse
import random

import numpy as np
import torch

from exp.exp_sup import Exp_All_Task
from utils.ddp import init_distributed_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Union supervised multi-task training")
    parser.add_argument("--task_name", type=str, default="ALL_task")
    parser.add_argument("--is_training", type=int, required=True, default=1)
    parser.add_argument("--model_id", type=str, required=True, default="test")
    parser.add_argument("--model", type=str, required=True, default="Union")
    parser.add_argument("--data", type=str, default="All")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--task_data_config_path", type=str, default="data_provider/multi_task.yaml")
    parser.add_argument("--subsample_pct", type=float, default=None)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--dist_url", default="env://", type=str)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--acc_it", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--lradj", type=str, default="supervised")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--pretrained_weight", type=str, default=None)
    parser.add_argument("--debug", type=str, default="disabled")
    parser.add_argument("--project_name", type=str, default="union-multitask")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--fix_seed", type=int, default=None)
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=1.0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument("--zero_shot_forecasting_new_length", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    init_distributed_mode(args)
    if args.fix_seed is not None:
        random.seed(args.fix_seed)
        torch.manual_seed(args.fix_seed)
        np.random.seed(args.fix_seed)
    print("Args in experiment:")
    print(args)
    for ii in range(args.itr):
        setting = f"{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}_dm{args.d_model}_el{args.e_layers}_{args.des}_{ii}"
        exp = Exp_All_Task(args)
        if args.is_training:
            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
        else:
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting, load_pretrain=True)


if __name__ == "__main__":
    main()
