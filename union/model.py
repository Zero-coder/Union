from dataclasses import dataclass
from typing import Dict, Literal

import torch
from torch import nn
import torch.nn.functional as F


TaskName = Literal["forecast", "impute", "anomaly", "classify"]


@dataclass
class UnionConfig:
    seq_len: int
    pred_len: int
    num_vars: int
    num_classes: int = 0
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    depth: int = 3
    n_heads: int = 4
    num_experts: int = 4
    top_k: int = 2
    dropout: float = 0.1


def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    if seq_len < patch_len:
        return 1
    return (seq_len - patch_len) // stride + 1


class SeriesTokenizer(nn.Module):
    def __init__(self, config: UnionConfig):
        super().__init__()
        self.config = config
        self.num_patches = _num_patches(config.seq_len, config.patch_len, config.stride)
        self.patch_proj = nn.Linear(config.patch_len * config.num_vars, config.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, config.d_model))
        self.prompt_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.gen_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.norm = nn.LayerNorm(config.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for token in (self.position, self.prompt_token, self.gen_token, self.cls_token):
            nn.init.normal_(token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, length, variables]
        if x.size(1) < self.config.patch_len:
            pad = self.config.patch_len - x.size(1)
            x = F.pad(x, (0, 0, 0, pad))
        patches = x.unfold(dimension=1, size=self.config.patch_len, step=self.config.stride)
        patches = patches.contiguous().flatten(start_dim=2)
        series_tokens = self.patch_proj(patches) + self.position[:, : patches.size(1)]
        batch = x.size(0)
        prefix = torch.cat(
            [
                self.prompt_token.expand(batch, -1, -1),
                self.gen_token.expand(batch, -1, -1),
                self.cls_token.expand(batch, -1, -1),
            ],
            dim=1,
        )
        return self.norm(torch.cat([prefix, series_tokens], dim=1))


class FrequencyAwareInteraction(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.channel = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.freq_scale = nn.Parameter(torch.ones(d_model))
        self.temporal = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        channel_update = self.channel(tokens)
        spectrum = torch.fft.rfft(tokens.float(), dim=1)
        spectrum = spectrum * self.freq_scale.view(1, 1, -1)
        temporal_update = torch.fft.irfft(spectrum, n=tokens.size(1), dim=1).to(tokens.dtype)
        temporal_update = self.temporal(temporal_update)
        update = channel_update + temporal_update
        return tokens + torch.sigmoid(self.gate) * update


class TCILinearExpert(nn.Module):
    def __init__(self, d_model: int, max_tokens: int, dropout: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_tokens, max_tokens))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def _resize_weight(self, length: int) -> torch.Tensor:
        weight = self.weight.unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(weight, size=(length, length), mode="bilinear", align_corners=False)
        resized = resized.squeeze(0).squeeze(0)
        idx = torch.arange(length, device=resized.device)
        distance = (idx[:, None] - idx[None, :]).abs().float()
        kernel = torch.exp(-distance / max(length / 8, 1))
        return resized * kernel

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        transform = self._resize_weight(tokens.size(1))
        mixed = torch.einsum("ij,bjd->bid", transform, tokens)
        return self.mlp(mixed)


class TCIMoE(nn.Module):
    def __init__(self, d_model: int, max_tokens: int, num_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.top_k = min(top_k, num_experts)
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList(
            [TCILinearExpert(d_model, max_tokens, dropout) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=1)
        scores = self.router(pooled)
        weights, indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        expert_outputs = torch.stack([expert(tokens) for expert in self.experts], dim=1)
        gather_index = indices[:, :, None, None].expand(-1, -1, tokens.size(1), tokens.size(2))
        selected = torch.gather(expert_outputs, dim=1, index=gather_index)
        return (selected * weights[:, :, None, None]).sum(dim=1)


class UnionBlock(nn.Module):
    def __init__(self, config: UnionConfig, max_tokens: int):
        super().__init__()
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            config.d_model,
            config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.freq = FrequencyAwareInteraction(config.d_model, config.dropout)
        self.moe_norm = nn.LayerNorm(config.d_model)
        self.moe = TCIMoE(
            config.d_model,
            max_tokens=max_tokens,
            num_experts=config.num_experts,
            top_k=config.top_k,
            dropout=config.dropout,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(tokens)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        tokens = tokens + attn_output
        tokens = self.freq(tokens)
        return tokens + self.moe(self.moe_norm(tokens))


class UnionModel(nn.Module):
    def __init__(self, config: UnionConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SeriesTokenizer(config)
        max_tokens = self.tokenizer.num_patches + 3
        self.blocks = nn.ModuleList([UnionBlock(config, max_tokens) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.d_model)
        self.generative_head = nn.Linear(config.d_model, config.pred_len * config.num_vars)
        self.reconstruction_head = nn.Linear(config.d_model, config.seq_len * config.num_vars)
        self.classification_head = nn.Linear(config.d_model, max(config.num_classes, 1))

    def forward(self, x: torch.Tensor, task: TaskName = "forecast") -> Dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        gen_token = tokens[:, 1]
        cls_token = tokens[:, 2]

        outputs: Dict[str, torch.Tensor] = {}
        if task == "forecast":
            y = self.generative_head(gen_token).view(x.size(0), self.config.pred_len, self.config.num_vars)
            outputs["forecast"] = y
        elif task in {"impute", "anomaly"}:
            y = self.reconstruction_head(gen_token).view(x.size(0), self.config.seq_len, self.config.num_vars)
            outputs["reconstruction"] = y
            if task == "anomaly":
                outputs["score"] = (y - x[:, : self.config.seq_len]).pow(2).mean(dim=-1)
        elif task == "classify":
            outputs["logits"] = self.classification_head(cls_token)
        else:
            raise ValueError(f"Unsupported task: {task}")
        return outputs
