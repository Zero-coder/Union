from dataclasses import dataclass
from typing import Literal, Tuple

import torch
from torch.utils.data import Dataset


TaskName = Literal["forecast", "impute", "anomaly", "classify"]


@dataclass
class SyntheticConfig:
    task: TaskName
    seq_len: int
    pred_len: int
    num_vars: int
    num_classes: int
    size: int = 512


class SyntheticTimeSeries(Dataset):
    def __init__(self, config: SyntheticConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        total_len = self.config.seq_len + self.config.pred_len
        time = torch.linspace(0, 8, total_len)
        phases = torch.rand(self.config.num_vars, generator=generator) * 3.14
        freqs = torch.randint(1, 5, (self.config.num_vars,), generator=generator).float()
        signal = torch.stack([torch.sin(time * freq + phase) for freq, phase in zip(freqs, phases)], dim=-1)
        signal = signal + 0.05 * torch.randn(signal.shape, generator=generator)

        x = signal[: self.config.seq_len]
        if self.config.task == "forecast":
            y = signal[self.config.seq_len :]
        elif self.config.task in {"impute", "anomaly"}:
            y = x.clone()
        else:
            y = torch.tensor(index % max(self.config.num_classes, 1), dtype=torch.long)
        return x.float(), y
