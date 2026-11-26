"""Guarded EML operator and projection-head modules for image SSL.

EML primitive: eml(x, y) = exp(x) - log(y).
Guarded form for stable training: exp(clamp(x, max=exp_clip)) - log(softplus(y) + eps).
"""
from __future__ import annotations

import torch
from torch import nn


class GuardedEML(nn.Module):
    """eml(x, y) = exp(clamp(x, max=exp_clip)) - log(softplus(y) + eps).

    softplus keeps the log argument strictly positive without hard clipping the
    gradient like ReLU would; clamp on the exp side caps overflow under fp16/bf16.
    """

    def __init__(self, exp_clip: float = 6.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.exp_clip = exp_clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_safe = torch.clamp(x, max=self.exp_clip)
        y_pos = torch.nn.functional.softplus(y) + self.eps
        return torch.exp(x_safe) - torch.log(y_pos)


class EMLActivation(nn.Module):
    """P2: EML as a drop-in ReLU replacement in ResNet blocks.

    Splits input channels in half → (x, y), applies GuardedEML, concatenates.
    Input and output have the same channel count. Channel dim must be even.
    """

    def __init__(self, exp_clip: float = 6.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.eml = GuardedEML(exp_clip, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        assert c % 2 == 0, f"EMLActivation requires even channels, got {c}"
        x_left, x_right = x[:, : c // 2], x[:, c // 2 :]
        return torch.cat([self.eml(x_left, x_right), self.eml(x_right, x_left)], dim=1)


class EMLProjectionHead(nn.Module):
    """SimCLR/BYOL-style projection head with one EML cell instead of an MLP.

    Architecture: Linear(in -> 2 * hidden) -> split -> EML -> Linear(hidden -> out).

    Splitting one Linear into the (x, y) operands lets EML act as the nonlinearity;
    parameter count stays close to the standard `Linear-ReLU-Linear` baseline.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.fc_split = nn.Linear(in_dim, 2 * hidden_dim)
        self.eml = GuardedEML()
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        xy = self.fc_split(h)
        x, y = xy[:, : self.hidden_dim], xy[:, self.hidden_dim :]
        z = self.eml(x, y)
        return self.fc_out(z)


class MLPProjectionHead(nn.Module):
    """Baseline SimCLR projection head: Linear -> ReLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(h)))


class EMLPhotometricAugment(nn.Module):
    """Guarded EML photometric view-generation transform (codex EML-Aug, ported).

    Applies an EML-style exp/log perturbation to image tensors in [0, 1]:
        a, b ~ U[1-strength, 1+strength]
        eml(a*centered, b*centered + 1) → renormalized → mixed back with original.
    Probability `p` controls how often the perturbation fires per call.

    Designed to be inserted into a torchvision Compose AFTER ToTensor and BEFORE
    Normalize, applied independently per view (standard SimCLR two-view setup).
    """

    def __init__(self, strength: float = 0.35, p: float = 0.8, eps: float = 1e-4) -> None:
        super().__init__()
        self.strength = strength
        self.p = p
        self.eps = eps

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return image
        centered = image - 0.5
        a = 1.0 + (torch.rand(()) * 2.0 - 1.0) * self.strength
        b = 1.0 + (torch.rand(()) * 2.0 - 1.0) * self.strength
        left = (a * centered).clamp(-2.0, 2.0)
        right = torch.nn.functional.softplus(b * centered + 1.0) + self.eps
        transformed = torch.exp(left) - torch.log(right)
        lo = transformed.amin(dim=(-2, -1), keepdim=True)
        hi = transformed.amax(dim=(-2, -1), keepdim=True)
        transformed = (transformed - lo) / (hi - lo + self.eps)
        mix = torch.rand(()) * self.strength
        return ((1.0 - mix) * image + mix * transformed).clamp(0.0, 1.0)


class SignedPowerProjectionHead(nn.Module):
    """Clipped polynomial high-gain probe followed by a learned linear layer.

    The hidden pre-power input is clipped to ``[-magnitude_clip, magnitude_clip]``
    before applying ``sign(z) * abs(z)**power``.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128,
                 power: float = 3.0, magnitude_clip: float = 20.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.power = power
        self.magnitude_clip = magnitude_clip

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc1(h)
        z = torch.clamp(z, min=-self.magnitude_clip, max=self.magnitude_clip)
        z = torch.sign(z) * z.abs().pow(self.power)
        return self.fc2(z)


class TanhProjectionHead(nn.Module):
    """Saturating hidden-core control: Linear -> Tanh -> Linear.

    Tanh bounds the hidden activation, but the learned output layer does not
    impose a fixed bound or moment on the final projection.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.tanh(self.fc1(h)))


class BridgedEMLProjectionHead(nn.Module):
    """B-EML: hidden EML activation followed by affine LayerNorm.

    Identical to EMLProjectionHead except a LayerNorm is inserted after the EML
    cell and before the learned output projection. LayerNorm conditions the
    hidden activation; it does not guarantee a fixed final output moment.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.fc_split = nn.Linear(in_dim, 2 * hidden_dim)
        self.eml = GuardedEML()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        xy = self.fc_split(h)
        x, y = xy[:, : self.hidden_dim], xy[:, self.hidden_dim :]
        z = self.eml(x, y)
        z = self.norm(z)
        return self.fc_out(z)


class ScaleNormProjectionHead(nn.Module):
    """Fixed per-sample norm control.

    Applies Linear -> ReLU -> Linear, row normalizes the result, and multiplies
    it by ``scale``. This fixes each sample norm; it does not enforce isotropic
    covariance or full rank.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.scale = float(scale)
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.fc2(self.relu(self.fc1(h)))
        z = torch.nn.functional.normalize(z, dim=1)
        return self.scale * z


class FixedProfileScaleNormProjectionHead(nn.Module):
    """Fixed coordinate-profile control at a constant per-sample norm.

    A deterministic geometric coordinate profile is applied before row
    normalization and multiplication by ``scale``. The intervention changes a
    fixed pre-normalization profile while keeping each final sample norm fixed;
    it does not directly measure learned output anisotropy.
    """
    def __init__(self, in_dim, hidden_dim=512, out_dim=128, power=1.0, scale=16.0, eps=1e-8):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.power = float(power)
        self.scale = float(scale)
        self.eps = float(eps)
        j = torch.arange(out_dim, dtype=torch.float32)
        w = float(power) ** (j / max(1, out_dim - 1))
        w = w / w.pow(2).mean().clamp_min(1e-12).sqrt()
        self.register_buffer("w", w)
    def forward(self, h):
        z = self.fc2(self.relu(self.fc1(h)))
        z = self.w * z
        z = torch.nn.functional.normalize(z, dim=1, eps=self.eps)
        return self.scale * z


def build_projection_head(kind: str, in_dim: int, hidden_dim: int = 512, out_dim: int = 128,
                          power: float = 3.0) -> nn.Module:
    if kind == "mlp":
        return MLPProjectionHead(in_dim, hidden_dim, out_dim)
    if kind == "eml":
        return EMLProjectionHead(in_dim, hidden_dim, out_dim)
    if kind == "signed_power":
        return SignedPowerProjectionHead(in_dim, hidden_dim, out_dim, power=power)
    if kind == "tanh":
        return TanhProjectionHead(in_dim, hidden_dim, out_dim)
    if kind == "beml":
        return BridgedEMLProjectionHead(in_dim, hidden_dim, out_dim)
    if kind == "scalenorm":
        return ScaleNormProjectionHead(in_dim, hidden_dim, out_dim, scale=power)
    if kind == "fixedprofile":
        return FixedProfileScaleNormProjectionHead(in_dim, hidden_dim, out_dim, power=power, scale=16.0)
    raise ValueError(f"unknown projection head kind: {kind!r}")


# Five controlled loss-facing head operators used by the stress test.
PROBE_HEADS = ("mlp", "eml", "signed_power", "tanh", "beml")


@torch.no_grad()
def projector_second_moment(head: nn.Module, h: torch.Tensor) -> float:
    """Mean per-channel second moment E[z^2] of the projector OUTPUT on batch h.

    Used to measure the feature-second-moment behavior that the paper's
    mechanism claim turns on. Returns +inf if the head produced non-finite
    output (i.e. it numerically diverged on this batch).
    """
    z = head(h)
    if not torch.isfinite(z).all():
        return float("inf")
    return z.pow(2).mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 512)
    print("=== projection head smoke test (CPU) ===")
    for kind in PROBE_HEADS:
        head = build_projection_head(kind, in_dim=512, hidden_dim=256, out_dim=128)
        z = head(x)
        n_params = sum(p.numel() for p in head.parameters())
        sm = projector_second_moment(head, x)
        print(f"{kind:>13}: out {tuple(z.shape)}, finite={torch.isfinite(z).all().item()}, "
              f"E[z^2]={sm:.4g}, params={n_params}")
    # Stress test: push pre-activations large to expose unbounded-amplifier growth.
    print("=== second-moment under scaled input (x10) -- amplifier signature ===")
    xb = x * 10.0
    for kind in PROBE_HEADS:
        head = build_projection_head(kind, in_dim=512, hidden_dim=256, out_dim=128)
        print(f"{kind:>13}: E[z^2]={projector_second_moment(head, xb):.4g}")
