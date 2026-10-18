"""Convenience launcher for a file-backed single-task training configuration."""
import argparse
from pathlib import Path

import yaml

from exp.exp_sup import Exp_All_Task
from run import build_parser, seed_everything, validate_args


def train(config):
    if "task_data_config_path" not in config:
        raise ValueError("Training config must set task_data_config_path to a YAML with real dataset paths. Use run.py directly for CLI training.")
    parser = build_parser()
    args = parser.parse_args(["--is_training", "1", "--model_id", str(config.get("model_id", "single_task"))])
    known = set(vars(args))
    aliases = {"seed": "fix_seed", "depth": "e_layers"}
    for key, value in config.items():
        key = aliases.get(key, key)
        if key not in known:
            raise ValueError("Unknown training option %r. Put dataset dimensions and paths in task_data_config_path." % key)
        setattr(args, key, value)
    validate_args(args)
    if args.itr != 1:
        raise ValueError("Use run.py --itr for repeated runs.")
    seed_everything(args.fix_seed)
    exp = Exp_All_Task(args)
    setting = args.model_id
    if args.is_training:
        exp.train(setting)
        return exp.test(setting)
    return exp.test(setting, load_pretrain=True)


def main():
    parser = argparse.ArgumentParser(description="Train Union from a file-backed dataset configuration.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if isinstance(config, dict) and "task_dataset" in config:
        config = {"task_data_config_path": str(args.config), "model_id": args.config.stem}
    train(config)


if __name__ == "__main__":
    main()
