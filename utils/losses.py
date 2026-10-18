import torch
from torch import nn


def masked_mse(pred, target, missing_mask):
    """Average squared error over hidden elements only; True means hidden."""
    if pred.shape != target.shape or missing_mask.shape != target.shape:
        raise ValueError("Prediction, target and missing mask must have identical shapes.")
    count = missing_mask.sum()
    if count.item() == 0:
        raise ValueError("The missing mask must select at least one element.")
    return ((pred - target).square() * missing_mask).sum() / count


class UnifiedMaskRecLoss(nn.Module):
    def forward(self, model_output, target, mask=None):
        pred = model_output.get("reconstruction")
        if pred is None:
            pred = model_output.get("forecast")
        if pred is None:
            raise ValueError("UnifiedMaskRecLoss expects reconstruction or forecast output.")
        if pred.shape != target.shape:
            raise ValueError("Reconstruction and target shapes must match exactly.")
        return {"loss": masked_mse(pred, target, mask) if mask is not None else (pred - target).square().mean()}


def mape_loss():
    return lambda pred, true: (torch.abs(pred - true) / true.abs().clamp_min(1e-5)).mean()


def smape_loss():
    return lambda pred, true: (2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true)).clamp_min(1e-5)).mean()


def mase_loss():
    raise NotImplementedError("MASE requires an in-sample seasonal naive denominator; use an explicit dataset-aware implementation.")
