"""Shared photometric corruption ops for codex linear-probe evaluators.

Extracted from `image_linear_probe.py` so the CIFAR and IN100 evaluators apply
the *exact same* tensor-space perturbation definitions (so the two robustness
tables in the paper are directly comparable).

Tensor convention: input is float in [0, 1], shape (..., C, H, W), pre-Normalize.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


CORRUPTION_CHOICES = ("none", "brightness", "contrast", "gamma", "gaussian_noise")


class TensorCorruption(nn.Module):
    """Tensor-space corruptions for codex robustness probes.

    These definitions MUST stay byte-identical to the CIFAR evaluator so that
    CIFAR and IN100 numbers in the paper are directly comparable.
    """

    def __init__(self, mode: str, severity: float) -> None:
        super().__init__()
        self.mode = mode
        self.severity = severity

    def forward(self, image: Tensor) -> Tensor:
        if self.mode == "none" or self.severity == 0:
            return image
        if self.mode == "brightness":
            return (image + self.severity).clamp(0.0, 1.0)
        if self.mode == "contrast":
            mean = image.mean(dim=(-2, -1), keepdim=True)
            return (mean + (1.0 + self.severity) * (image - mean)).clamp(0.0, 1.0)
        if self.mode == "gamma":
            gamma = max(0.1, 1.0 + self.severity)
            return image.clamp(0.0, 1.0).pow(gamma)
        if self.mode == "gaussian_noise":
            return (image + torch.randn_like(image) * self.severity).clamp(0.0, 1.0)
        raise ValueError(f"unknown corruption: {self.mode}")
