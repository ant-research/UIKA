import torch
import torch.nn as nn


__all__ = ['OpacityEntropyLoss']


class OpacityEntropyLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, opacity: torch.Tensor) -> torch.Tensor:
        r"""
        Penalize semi-transparent Gaussian opacity by minimizing binary entropy.

        Args:
            opacity: (..., 1), activated opacity in [0, 1].

        Returns:
            Mean binary entropy over all Gaussian opacity values.
        """
        alpha = opacity.float().clamp(self.eps, 1.0 - self.eps)
        entropy = -(alpha * torch.log(alpha) + (1.0 - alpha) * torch.log(1.0 - alpha))
        return entropy.mean()
