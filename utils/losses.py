import torch
from torch import nn


class UnifiedMaskRecLoss(nn.Module):
    def forward(self, model_output, target, mask=None):
        pred = model_output.get("reconstruction") or model_output.get("forecast")
        if pred is None:
            raise ValueError("UnifiedMaskRecLoss expects reconstruction or forecast output.")
        pred = pred[:, : target.size(1), : target.size(2)]
        target = target[:, : pred.size(1), : pred.size(2)]
        loss = (pred - target).pow(2)
        if mask is not None:
            loss = loss * mask[:, : pred.size(1), : pred.size(2)]
        return {"loss": loss.mean()}


def mape_loss():
    return lambda pred, true: (torch.abs((true - pred) / true.clamp_min(1e-5))).mean()


def smape_loss():
    return lambda pred, true: (2 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true)).clamp_min(1e-5)).mean()


def mase_loss():
    return nn.L1Loss()
