"""Adapter from dataset/task configurations to the shared Union model."""

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from union.model import UnionConfig, UnionModel


def _max_from_configs(task_data_config_list: Optional[List[List[Any]]], key: str, default: int) -> int:
    values = []
    for _, config in task_data_config_list or []:
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            values.append(value)
    return max(values) if values else default


class Model(nn.Module):
    def __init__(self, args, task_data_config_list=None, pretrain: bool = False):
        super().__init__()
        self.args = args
        self.task_data_config_list = task_data_config_list or []
        self.pretrain = pretrain
        reference_length = getattr(args, "reference_seq_len", None)
        if reference_length is not None and reference_length <= 0:
            raise ValueError("reference_seq_len must be positive when specified")
        config = UnionConfig(
            seq_len=reference_length or _max_from_configs(self.task_data_config_list, "seq_len", 96),
            pred_len=_max_from_configs(self.task_data_config_list, "pred_len", 96),
            num_vars=_max_from_configs(self.task_data_config_list, "enc_in", 1),
            num_classes=_max_from_configs(self.task_data_config_list, "num_class", 1),
            patch_len=args.patch_len,
            stride=args.stride,
            d_model=args.d_model,
            depth=args.e_layers,
            n_heads=args.n_heads,
            num_experts=getattr(args, "num_experts", 4),
            top_k=getattr(args, "top_k", 2),
            dropout=args.dropout,
            prompt_num=getattr(args, "prompt_num", 10),
            kernel_tau=getattr(args, "kernel_tau", 0.2),
            max_freq_modes=getattr(args, "max_freq_modes", 64),
        )
        self.model = UnionModel(config)
        self.dataset_ids = []
        for task_key, spec in self.task_data_config_list:
            dataset_id = str(spec.get("dataset") or spec.get("dataset_name") or task_key)
            self.dataset_ids.append(dataset_id)
            self.model.register_dataset(dataset_id)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        task_id: int = 0,
        task_name: Optional[str] = None,
        enable_mask: bool = False,
        mask: Optional[torch.Tensor] = None,
        **_: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        if enable_mask and mask is None:
            raise ValueError("enable_mask requires an explicit observed-value mask")
        if self.task_data_config_list:
            if not 0 <= task_id < len(self.task_data_config_list):
                raise ValueError(f"Unknown task_id: {task_id}")
            spec = self.task_data_config_list[task_id][1]
            dataset_id = self.dataset_ids[task_id]
            task_name = task_name or spec.get("task_name")
        else:
            spec, dataset_id = {}, "default"
        return self.model(
            x_enc, task=self._map_task(task_name), dataset_id=dataset_id,
            pred_len=spec.get("pred_len") or None,
            num_classes=spec.get("num_class") or None,
            mask=mask,
        )

    def _map_task(self, task_name: Optional[str]) -> str:
        if self.pretrain or task_name is None:
            return "impute"
        names = {
            "forecast": "forecast", "long_term_forecast": "forecast", "short_term_forecast": "forecast",
            "impute": "impute", "imputation": "impute", "anomaly": "anomaly",
            "anomaly_detection": "anomaly", "classify": "classify", "classification": "classify",
        }
        if task_name not in names:
            raise ValueError(f"Unsupported task_name: {task_name}")
        return names[task_name]
