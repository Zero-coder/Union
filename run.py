"""Supervised multi-task training with file-backed data and saved checkpoints."""
import argparse
import os
import random

import numpy as np
import torch

from exp.exp_sup import Exp_All_Task


def build_parser():
    parser = argparse.ArgumentParser(description="Union supervised multi-task training")
    parser.add_argument("--task_name", default="ALL_task")
    parser.add_argument("--is_training", type=int, choices=[0, 1], required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--model", default="Union")
    parser.add_argument("--data", default="All", help="Experiment identifier; dataset files are selected by YAML.")
    parser.add_argument("--features", choices=["M", "S"], default="M")
    parser.add_argument("--target", default="OT")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--task_data_config_path", default="configs/forecast_csv.yaml")
    parser.add_argument("--subsample_pct", type=float, default=None, help="Fraction of training windows/samples in (0,1].")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--prompt_tune_epoch", type=int, default=0)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--acc_it", type=int, default=1, help="Number of batches per optimizer update.")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--des", default="experiment")
    parser.add_argument("--lradj", default="supervised")
    parser.add_argument("--clip_grad", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--checkpoints", default="./checkpoints/")
    parser.add_argument("--pretrained_weight", default=None)
    parser.add_argument("--debug", default="disabled")
    parser.add_argument("--project_name", default="union-multitask", help="Metadata recorded in the checkpoint.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--prompt_num", type=int, default=10)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--kernel_tau", type=float, default=0.2)
    parser.add_argument("--max_freq_modes", type=int, default=64)
    parser.add_argument("--reference_seq_len", type=int, default=None, help="Base weight length used by a source checkpoint when target input lengths differ.")
    parser.add_argument("--fix_seed", type=int, default=2021)
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=1.0, help="Percentile tail size for validation-based anomaly threshold.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max_offset", type=int, default=0)
    parser.add_argument("--zero_shot_forecasting_new_length", default=None)
    return parser


def validate_args(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or args.local_rank != 0 or args.dist_url != "env://":
        raise ValueError("Only single-process training is supported; distributed options are not implemented.")
    unsupported = {"prompt_tune_epoch": 0, "offset": 0, "max_offset": 0,
                   "zero_shot_forecasting_new_length": None, "debug": "disabled", "freq": "h"}
    for name, default in unsupported.items():
        if getattr(args, name) != default:
            raise ValueError("--%s=%r is not implemented; use its default %r." % (name, getattr(args, name), default))
    if args.lradj not in {"supervised", "constant", "cosine"}:
        raise ValueError("lradj supports supervised, constant and cosine; prompt-only tuning is not implemented.")
    if args.pretrained_weight == "auto":
        raise ValueError("Provide an explicit pretrained_weight checkpoint path; auto lookup is not implemented.")
    for name in ["itr", "batch_size", "acc_it", "d_model", "n_heads", "e_layers", "patch_len", "stride", "prompt_num", "num_experts", "top_k", "max_freq_modes"]:
        if getattr(args, name) < 1:
            raise ValueError("--%s must be positive." % name)
    if args.top_k > args.num_experts or args.d_model % args.n_heads:
        raise ValueError("top_k must not exceed num_experts, and d_model must be divisible by n_heads.")
    if args.stride > args.patch_len or args.fix_seed < 0:
        raise ValueError("stride must not exceed patch_len; fix_seed must be nonnegative.")
    if args.reference_seq_len is not None and args.reference_seq_len < 1:
        raise ValueError("reference_seq_len must be positive when provided.")
    if args.is_training and args.train_epochs < 1:
        raise ValueError("train_epochs must be positive for training.")
    if not 0 <= args.warmup_epochs <= args.train_epochs or args.num_workers < 0:
        raise ValueError("Invalid warmup_epochs or num_workers.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or (args.min_lr is not None and not 0 <= args.min_lr <= args.learning_rate):
        raise ValueError("Invalid optimizer learning rate or weight decay.")
    if args.clip_grad is not None and args.clip_grad <= 0:
        raise ValueError("clip_grad must be positive.")
    if not 0 < args.mask_rate < 1 or not 0 < args.anomaly_ratio < 100:
        raise ValueError("mask_rate must be in (0,1); anomaly_ratio must be in (0,100).")
    if args.subsample_pct is not None and not 0 < args.subsample_pct <= 1:
        raise ValueError("subsample_pct must be in (0,1].")
    if not 0 <= args.dropout < 1 or args.kernel_tau <= 0:
        raise ValueError("dropout must be in [0,1); kernel_tau must be positive.")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def experiment_setting(args, repetition):
    return "%s_%s_%s_dm%d_el%d_%s_%d" % (args.task_name, args.model_id, args.model, args.d_model, args.e_layers, args.des, repetition)


def main():
    args = build_parser().parse_args()
    validate_args(args)
    base_seed = args.fix_seed
    for repetition in range(args.itr):
        args.fix_seed = base_seed + repetition
        seed_everything(args.fix_seed)
        setting = experiment_setting(args, repetition)
        print("Experiment %s; seed=%d" % (setting, args.fix_seed))
        exp = Exp_All_Task(args)
        if args.is_training:
            exp.train(setting)
            exp.test(setting)
        else:
            exp.test(setting, load_pretrain=True)


if __name__ == "__main__":
    main()
