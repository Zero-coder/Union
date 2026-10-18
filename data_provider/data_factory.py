"""Build deterministic loaders from explicitly configured dataset files."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from union.data import load_dataset


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def data_provider(args, config, flag, ddp=False):
    if ddp:
        raise NotImplementedError("This entry point supports one training process. Distributed training is not implemented.")
    config = dict(config)
    for key, fallback in [("features", "M"), ("target", "OT"), ("freq", "h")]:
        config.setdefault(key, getattr(args, key, fallback))
    seed = int(getattr(args, "fix_seed", 0) or 0)
    dataset = load_dataset(config, flag, getattr(args, "subsample_pct", None), seed)
    batch_size = int(config.get("max_batch") or args.batch_size)
    if batch_size < 1:
        raise ValueError("batch_size/max_batch must be positive.")
    generator = torch.Generator().manual_seed(seed + {"train": 0, "val": 1, "test": 2}[flag])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=flag == "train",
                        num_workers=args.num_workers, drop_last=False,
                        generator=generator, worker_init_fn=_seed_worker)
    return dataset, loader
