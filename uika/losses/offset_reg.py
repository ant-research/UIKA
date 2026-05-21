import torch
import torch.nn as nn


__all__ = ['OffsetReg']


class OffsetReg(nn.Module):
    def __init__(self, sigma=0.0):
        super().__init__()
        self.sigma = sigma
        self.mse_loss = nn.MSELoss()

    def forward(self, pred_offset):
        r"""
        Args:
            pred_offset: (B, N, 3)

        Returns:
            loss
        """
        target = torch.full_like(pred_offset, self.sigma, device=pred_offset.device)
        return self.mse_loss(pred_offset, target)
