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
    """Compatibility wrapper for UniTS-style experiment code."""

    def __init__(self, args, task_data_config_list=None, pretrain: bool = False):
        super().__init__()
        self.args = args
        self.task_data_config_list = task_data_config_list or []
        self.pretrain = pretrain
        config = UnionConfig(
            seq_len=_max_from_configs(self.task_data_config_list, "seq_len", 96),
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
        )
        self.model = UnionModel(config)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: Optional[torch.Tensor] = None,
        task_id: int = 0,
        task_name: Optional[str] = None,
        enable_mask: bool = False,
        **_: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        task = self._map_task(task_name)
        x_enc = self._adapt_input(x_enc)
        return self.model(x_enc, task=task)

    def _adapt_input(self, x_enc: torch.Tensor) -> torch.Tensor:
        target_vars = self.model.config.num_vars
        if x_enc.size(-1) < target_vars:
            pad = target_vars - x_enc.size(-1)
            x_enc = torch.nn.functional.pad(x_enc, (0, pad))
        elif x_enc.size(-1) > target_vars:
            x_enc = x_enc[..., :target_vars]
        return x_enc

    def _map_task(self, task_name: Optional[str]) -> str:
        if self.pretrain or task_name is None:
            return "impute"
        if "forecast" in task_name:
            return "forecast"
        if "imputation" in task_name:
            return "impute"
        if "anomaly" in task_name:
            return "anomaly"
        if "classification" in task_name:
            return "classify"
        return "forecast"
