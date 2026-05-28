from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from union.data import SyntheticConfig, SyntheticTimeSeries


def _task_to_synthetic_name(task_name: str) -> str:
    if "forecast" in task_name:
        return "forecast"
    if "imputation" in task_name:
        return "impute"
    if "anomaly" in task_name:
        return "anomaly"
    if "classification" in task_name:
        return "classify"
    return "forecast"


def data_provider(args: Any, config: Dict[str, Any], flag: str, ddp: bool = False) -> Tuple[SyntheticTimeSeries, DataLoader]:
    task = _task_to_synthetic_name(config["task_name"])
    size = 512 if flag == "train" else 128
    dataset = SyntheticTimeSeries(
        SyntheticConfig(
            task=task,
            seq_len=int(config["seq_len"]),
            pred_len=int(config.get("pred_len", 0) or config["seq_len"]),
            num_vars=int(config.get("enc_in") or config.get("c_out") or 1),
            num_classes=int(config.get("num_class") or 1),
            size=size,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("max_batch") or args.batch_size),
        shuffle=flag == "train",
        num_workers=args.num_workers,
        drop_last=flag == "train",
    )
    return dataset, loader
