"""Union model components.

Implementation conventions: retain a variable axis in the hidden representation,
broadcast each dataset's prompt over variables, and share a patch decoder across
sequence-output tasks. These choices make variable lengths and channel counts
explicit; they do not imply compatibility with earlier checkpoints.
"""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

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
    prompt_num: int = 10
    kernel_tau: float = 0.2
    max_freq_modes: int = 64

    def __post_init__(self) -> None:
        for name in (
            "seq_len", "num_vars", "patch_len", "stride", "d_model", "depth",
            "n_heads", "num_experts", "top_k", "prompt_num", "max_freq_modes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.pred_len < 0 or self.num_classes < 0:
            raise ValueError("pred_len and num_classes must be non-negative")
        if self.stride > self.patch_len:
            raise ValueError("stride must not exceed patch_len (all time steps must be covered)")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k must not exceed num_experts")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not 0 < self.kernel_tau < float("inf"):
            raise ValueError("kernel_tau must be finite and positive")


def _num_patches(seq_len: int, patch_len: int, stride: int) -> int:
    """Include a final padded patch instead of dropping trailing observations."""
    return max(1, (max(seq_len - patch_len, 0) + stride - 1) // stride + 1)


class SeriesTokenizer(nn.Module):
    def __init__(self, config: UnionConfig):
        super().__init__()
        self.config = config
        self.num_patches = _num_patches(config.seq_len, config.patch_len, config.stride)
        self.patch_proj = nn.Linear(config.patch_len, config.d_model)
        self.position = nn.Parameter(torch.empty(1, self.num_patches, config.d_model))
        self.prompt_tokens = nn.ParameterDict()
        self.gen_token = nn.Parameter(torch.empty(1, 1, 1, config.d_model))
        self.cls_token = nn.Parameter(torch.empty(1, 1, 1, config.d_model))
        self.norm = nn.LayerNorm(config.d_model)
        for token in (self.position, self.gen_token, self.cls_token):
            nn.init.normal_(token, std=0.02)
        self.register_dataset("default")

    @staticmethod
    def _dataset_key(dataset_id: str) -> str:
        if not isinstance(dataset_id, str) or not dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        # Reversible encoding avoids ParameterDict restrictions on periods.
        return "dataset_" + dataset_id.encode("utf-8").hex()

    def register_dataset(self, dataset_id: str) -> nn.Parameter:
        """Register new prompts before constructing an optimizer or loading weights."""
        key = self._dataset_key(dataset_id)
        if key not in self.prompt_tokens:
            prompt = nn.Parameter(self.position.new_empty(1, 1, self.config.prompt_num, self.config.d_model))
            nn.init.normal_(prompt, std=0.02)
            self.prompt_tokens[key] = prompt
        return self.prompt_tokens[key]

    def get_prompt(self, dataset_id: str) -> torch.Tensor:
        key = self._dataset_key(dataset_id)
        if key not in self.prompt_tokens:
            raise ValueError(f"Unknown dataset {dataset_id!r}; call register_dataset before creating the optimizer")
        return self.prompt_tokens[key]

    def forward(self, x: torch.Tensor, dataset_id: str = "default") -> torch.Tensor:
        # x: [batch, length, variables]; hidden layout: [batch, variables, tokens, d].
        batch, length, variables = x.shape
        count = _num_patches(length, self.config.patch_len, self.config.stride)
        padded_length = (count - 1) * self.config.stride + self.config.patch_len
        series = F.pad(x.transpose(1, 2), (0, padded_length - length))
        patches = series.unfold(-1, self.config.patch_len, self.config.stride)
        position = F.interpolate(
            self.position.transpose(1, 2), size=count, mode="linear", align_corners=False,
        ).transpose(1, 2)
        series_tokens = self.patch_proj(patches) + position[:, None]
        prompt = self.get_prompt(dataset_id).expand(batch, variables, -1, -1)
        cls = self.cls_token.expand(batch, variables, -1, -1)
        gen = self.gen_token.expand(batch, variables, -1, -1)
        return self.norm(torch.cat([prompt, cls, series_tokens, gen], dim=2))


class FrequencyAttentionEncoder(nn.Module):
    """Attention over selected frequency positions, shared by real/imaginary parts."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

    @staticmethod
    def _normalize(x: torch.Tensor, norm: nn.BatchNorm1d) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        if norm.training and x.shape[0] * x.shape[1] == 1:
            # A one-item, one-frequency batch has no batch variance estimate.
            normalized = F.batch_norm(
                channels_first, norm.running_mean, norm.running_var,
                norm.weight, norm.bias, training=False, eps=norm.eps,
            )
        else:
            normalized = norm(channels_first)
        return normalized.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update, _ = self.attn(x, x, x, need_weights=False)
        x = self._normalize(x + self.dropout(update), self.norm1)
        return self._normalize(x + self.dropout(self.ffn(x)), self.norm2)


class FrequencyAxisInteraction(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_modes: int):
        super().__init__()
        self.max_modes = max_modes
        self.encoder = FrequencyAttentionEncoder(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # Move the selected interaction axis just before the feature axis.
        rearranged = x.movedim(axis, -2)
        shape = rearranged.shape
        length = shape[-2]
        flattened = rearranged.reshape(-1, length, shape[-1])
        fft_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        spectrum = torch.fft.rfft(flattened.to(fft_dtype), dim=1)
        frequency_count = spectrum.size(1)
        if frequency_count <= self.max_modes:
            indices = torch.arange(frequency_count, device=x.device)
        else:
            # One index set per stage/forward pass, shared by all samples/slices.
            # Sampling also occurs in eval mode, following Appendix B.
            indices = torch.randperm(frequency_count, device=x.device)[:self.max_modes].sort().values
        selected = spectrum.index_select(1, indices)
        real = self.encoder(selected.real.to(x.dtype)).to(fft_dtype)
        imag = self.encoder(selected.imag.to(x.dtype)).to(fft_dtype)
        processed = torch.complex(real, imag)
        restored = torch.zeros_like(spectrum).index_copy(1, indices, processed)
        reconstructed = torch.fft.irfft(restored, n=length, dim=1).to(x.dtype)
        return reconstructed.reshape(shape).movedim(-2, axis)


class FrequencyAwareInteraction(nn.Module):
    def __init__(self, d_model: int, dropout: float, n_heads: int = 4, max_modes: int = 64):
        super().__init__()
        self.channel = FrequencyAxisInteraction(d_model, n_heads, dropout, max_modes)
        self.temporal = FrequencyAxisInteraction(d_model, n_heads, dropout, max_modes)
        self.projection = nn.Linear(d_model, d_model)
        # A bounded per-feature gate is the implementation convention here.
        self.gate = nn.Parameter(torch.zeros(d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        channel_update = self.channel(tokens, axis=1)
        temporal_update = self.temporal(channel_update, axis=2)
        return tokens + torch.sigmoid(self.gate) * self.projection(temporal_update)


class TCILinearExpert(nn.Module):
    def __init__(self, d_model: int, max_tokens: int, dropout: float, kernel_tau: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_tokens, max_tokens))
        self.register_buffer("kernel_tau", torch.tensor(float(kernel_tau)))
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        nn.init.xavier_uniform_(self.weight)

    def temporal_kernel(self, length: int) -> torch.Tensor:
        idx = torch.arange(length, device=self.weight.device, dtype=self.weight.dtype)
        distance = (idx[:, None] - idx[None, :]).abs()
        return torch.exp(-distance / (self.kernel_tau * max(length - 1, 1)))

    def _resize_weight(self, length: int) -> torch.Tensor:
        resized = F.interpolate(
            self.weight[None, None], size=(length, length), mode="bilinear", align_corners=False,
        )[0, 0]
        return resized * self.temporal_kernel(length)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,...jd->...id", self._resize_weight(tokens.size(-2)), tokens)
        return self.mlp(mixed)


class TCIMoE(nn.Module):
    def __init__(self, d_model: int, max_tokens: int, num_experts: int, top_k: int,
                 dropout: float, kernel_tau: float = 0.2):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts")
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            TCILinearExpert(d_model, max_tokens, dropout, kernel_tau) for _ in range(num_experts)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = tokens.mean(dim=tuple(range(1, tokens.ndim - 1)))
        scores = self.router(pooled)
        selected_scores, indices = torch.topk(scores, k=self.top_k, dim=-1)
        # Equivalent to softmax over all experts followed by selected-weight renormalization.
        weights = F.softmax(selected_scores, dim=-1)
        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            batch_indices, slots = torch.where(indices == expert_id)
            if batch_indices.numel() == 0:
                continue
            selected = expert(tokens.index_select(0, batch_indices))
            scale = weights[batch_indices, slots].reshape(-1, *([1] * (tokens.ndim - 1)))
            output = output.index_add(0, batch_indices, selected * scale)
        return output


class UnionBlock(nn.Module):
    def __init__(self, config: UnionConfig, max_tokens: int):
        super().__init__()
        self.freq = FrequencyAwareInteraction(
            config.d_model, config.dropout, config.n_heads, config.max_freq_modes,
        )
        self.moe_norm = nn.LayerNorm(config.d_model)
        self.moe = TCIMoE(
            config.d_model, max_tokens, config.num_experts, config.top_k,
            config.dropout, config.kernel_tau,
        )
        self.moe_gate = nn.Parameter(torch.zeros(config.d_model))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.freq(tokens)
        return tokens + torch.sigmoid(self.moe_gate) * self.moe(self.moe_norm(tokens))


class UnionModel(nn.Module):
    def __init__(self, config: UnionConfig):
        super().__init__()
        self.config = config
        self.tokenizer = SeriesTokenizer(config)
        max_tokens = self.tokenizer.num_patches + config.prompt_num + 2
        self.blocks = nn.ModuleList([UnionBlock(config, max_tokens) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.d_model)
        self.generative_head = nn.Linear(config.d_model, config.patch_len)
        self.classification_head = nn.Linear(config.d_model, max(config.num_classes, 1))

    def register_dataset(self, dataset_id: str) -> nn.Parameter:
        return self.tokenizer.register_dataset(dataset_id)

    def _decode_sequence(self, series: torch.Tensor, gen: torch.Tensor, length: int) -> torch.Tensor:
        """Shared decoder: GEN-conditioned series features, interpolation, overlap averaging."""
        batch, variables, _, width = series.shape
        count = _num_patches(length, self.config.patch_len, self.config.stride)
        features = (series + gen).reshape(batch * variables, -1, width).transpose(1, 2)
        features = F.interpolate(features, size=count, mode="linear", align_corners=False).transpose(1, 2)
        patches = self.generative_head(features).transpose(1, 2)
        padded_length = (count - 1) * self.config.stride + self.config.patch_len
        fold_args = dict(output_size=(1, padded_length), kernel_size=(1, self.config.patch_len),
                         stride=(1, self.config.stride))
        total = F.fold(patches, **fold_args)
        counts = F.fold(torch.ones_like(patches), **fold_args)
        decoded = (total / counts).reshape(batch, variables, padded_length)[..., :length]
        return decoded.transpose(1, 2)

    def forward(self, x: torch.Tensor, task: TaskName = "forecast", dataset_id: str = "default",
                pred_len: Optional[int] = None, num_classes: Optional[int] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if x.ndim != 3 or any(size < 1 for size in x.shape):
            raise ValueError("x must have non-empty shape [batch, length, variables]")
        if not x.is_floating_point():
            raise ValueError("x must be a floating-point tensor")
        if mask is not None:
            if mask.shape != x.shape:
                raise ValueError("mask must have the same shape as x; True means observed")
            x = torch.where(mask.to(device=x.device, dtype=torch.bool), x, torch.zeros_like(x))
        if not torch.isfinite(x).all():
            raise ValueError("x contains non-finite observed values")
        tokens = self.tokenizer(x, dataset_id)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        cls = tokens[:, :, self.config.prompt_num].mean(dim=1)
        series = tokens[:, :, self.config.prompt_num + 1:-1]
        gen = tokens[:, :, -1:]
        if task == "forecast":
            horizon = self.config.pred_len if pred_len is None else pred_len
            if horizon <= 0:
                raise ValueError("forecast pred_len must be positive")
            return {"forecast": self._decode_sequence(series, gen, horizon)}
        if task in {"impute", "anomaly"}:
            reconstructed = self._decode_sequence(series, gen, x.size(1))
            outputs = {"reconstruction": reconstructed}
            if task == "anomaly":
                outputs["score"] = (reconstructed - x).square().mean(dim=-1)
            return outputs
        if task == "classify":
            classes = self.config.num_classes if num_classes is None else num_classes
            if not 1 <= classes <= self.classification_head.out_features:
                raise ValueError("num_classes must fit the configured classification head")
            return {"logits": self.classification_head(cls)[:, :classes]}
        raise ValueError(f"Unsupported task: {task}")
