"""Shared building blocks: normalization, convolutions, speaker fusion, encoder/decoder."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLayerNorm(nn.Module):
    """Global layer norm (gLN) over channels and time, per sample.

    Statistics are computed in float32 so the layer is stable under bf16 autocast.
    """

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)."""
        xf = x.float()
        mean = xf.mean(dim=(1, 2), keepdim=True)
        var = (xf - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        y = (xf - mean) / torch.sqrt(var + self.eps)
        return (self.gamma * y + self.beta).to(x.dtype)


class DSConvBlock(nn.Module):
    """Depthwise-separable Conv1d -> PReLU -> gLN (keeps length for stride 1, halves for 2)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1,
                 dilation: int = 1) -> None:
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, stride=stride, padding=pad,
                                   dilation=dilation, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.act = nn.PReLU()
        self.norm = GlobalLayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.act(self.pointwise(self.depthwise(x))))


class FiLM(nn.Module):
    """Feature-wise linear modulation ``gamma(e) * x + beta(e)``.

    Weights are zero-initialised and the gamma bias is one, so the layer is exactly
    the identity at initialisation for any embedding.
    """

    def __init__(self, spk_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(spk_dim, channels)
        self.beta = nn.Linear(spk_dim, channels)
        for lin, b in ((self.gamma, 1.0), (self.beta, 0.0)):
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, b)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), e: (B, D)."""
        return self.gamma(e).unsqueeze(-1) * x + self.beta(e).unsqueeze(-1)


class MultFusion(nn.Module):
    """Multiplicative adaptation ``x * W e`` (TD-SpeakerBeam style)."""

    def __init__(self, spk_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Linear(spk_dim, channels)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return x * self.proj(e).unsqueeze(-1)


class ConcatFusion(nn.Module):
    """Concatenate the (time-tiled) embedding and project back with a 1x1 conv."""

    def __init__(self, spk_dim: int, channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(channels + spk_dim, channels, 1)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        e_t = e.unsqueeze(-1).expand(-1, -1, x.shape[-1]).to(x.dtype)
        return self.proj(torch.cat([x, e_t], dim=1))


def make_fusion(kind: str, spk_dim: int, channels: int) -> nn.Module:
    """Factory for ``film`` | ``mult`` | ``concat`` speaker fusion."""
    if kind == "film":
        return FiLM(spk_dim, channels)
    if kind == "mult":
        return MultFusion(spk_dim, channels)
    if kind == "concat":
        return ConcatFusion(spk_dim, channels)
    raise ValueError(f"Unknown fusion '{kind}'")


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation over channels (mean over time -> MLP -> sigmoid)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(),
                                nn.Linear(hidden, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x.mean(dim=-1)).unsqueeze(-1)


class MultiHeadAttention(nn.Module):
    """Pre-norm multi-head self-attention over time with a residual path.

    Tokens are the frames of ``x`` (B, C, T); queries/keys/values live in ``attn_dim``
    (< C keeps it cheap). The output projection is zero-initialised, so the layer is the
    identity at initialisation.
    """

    def __init__(self, channels: int, attn_dim: int = 256, heads: int = 4) -> None:
        super().__init__()
        if attn_dim % heads:
            raise ValueError(f"attn_dim {attn_dim} not divisible by heads {heads}")
        self.heads = heads
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, 3 * attn_dim)
        self.out = nn.Linear(attn_dim, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _split(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, heads, T, D / heads)."""
        Bn, T, D = t.shape
        return t.view(Bn, T, self.heads, D // self.heads).transpose(1, 2)

    def attend(self, h: torch.Tensor) -> torch.Tensor:
        """(B, T, C) normalised tokens -> (B, heads, T, D / heads) attention output."""
        q, k, v = (self._split(t) for t in self.qkv(h).chunk(3, dim=-1))
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, C, T)."""
        h = self.norm(x.transpose(1, 2))
        o = self.attend(h)
        o = o.transpose(1, 2).reshape(h.shape[0], h.shape[1], -1)
        return x + self.out(o).transpose(1, 2)


class DifferentialAttention(MultiHeadAttention):
    """Differential attention (Ye et al., 2024, "Differential Transformer") over time.

    Each head computes two softmax maps from split queries/keys and uses their difference
    ``(softmax(Q1 K1^T) - lambda * softmax(Q2 K2^T)) V``, which cancels attention mass both
    maps share (common-mode "noise"). ``lambda`` is re-parameterised as
    ``exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`` with ``lambda_init`` set by ``depth``; each
    head output is RMS-normalised and scaled by ``1 - lambda_init``.

    Uses ``heads // 2`` heads of twice the width, so parameters match
    :class:`MultiHeadAttention` with the same ``attn_dim`` and ``heads``.
    """

    def __init__(self, channels: int, attn_dim: int = 256, heads: int = 4, depth: int = 0) -> None:
        if heads % 2:
            raise ValueError(f"differential attention needs an even number of heads, got {heads}")
        super().__init__(channels, attn_dim, heads // 2)
        dh = attn_dim // self.heads // 2  # width of each of the two query/key halves
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lq1, self.lk1, self.lq2, self.lk2 = (
            nn.Parameter(torch.randn(dh) * 0.1) for _ in range(4))
        self.head_norm = nn.RMSNorm(2 * dh, eps=1e-5, elementwise_affine=False)

    def attend(self, h: torch.Tensor) -> torch.Tensor:
        q, k, v = (self._split(t) for t in self.qkv(h).chunk(3, dim=-1))
        q1, q2 = q.chunk(2, dim=-1)
        k1, k2 = k.chunk(2, dim=-1)
        lam = (torch.exp((self.lq1 * self.lk1).sum()) - torch.exp((self.lq2 * self.lk2).sum())
               + self.lambda_init)
        o = (F.scaled_dot_product_attention(q1, k1, v)
             - lam * F.scaled_dot_product_attention(q2, k2, v))
        return self.head_norm(o) * (1.0 - self.lambda_init)


def make_attention(kind: str, channels: int, se_reduction: int = 8, attn_dim: int = 256,
                   heads: int = 4, depth: int = 0) -> nn.Module:
    """Factory for the U-block bottleneck attention: ``se`` | ``mhsa`` | ``diff``."""
    if kind == "se":
        return ChannelAttention(channels, se_reduction)
    if kind == "mhsa":
        return MultiHeadAttention(channels, attn_dim, heads)
    if kind == "diff":
        return DifferentialAttention(channels, attn_dim, heads, depth)
    raise ValueError(f"Unknown attention '{kind}'")


class WaveEncoder(nn.Module):
    """Learned analysis filterbank: Conv1d(1, N, L, stride) + ReLU."""

    def __init__(self, N: int, L: int, stride: int) -> None:
        super().__init__()
        self.L, self.stride = L, stride
        self.conv = nn.Conv1d(1, N, L, stride=stride, bias=False)

    def pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Pad (B, T) so that frames tile the signal fully; returns padded (B, 1, T') and T."""
        T = x.shape[-1]
        edge = self.L - self.stride
        total = T + 2 * edge
        extra = (self.stride - (total - self.L) % self.stride) % self.stride
        return F.pad(x.unsqueeze(1), (edge, edge + extra)), T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T) -> (B, N, F); no length bookkeeping (used for enrollments)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return F.relu(self.conv(x))


class WaveDecoder(nn.Module):
    """Learned synthesis filterbank: ConvTranspose1d(N, 1, L, stride)."""

    def __init__(self, N: int, L: int, stride: int) -> None:
        super().__init__()
        self.edge = L - stride
        self.deconv = nn.ConvTranspose1d(N, 1, L, stride=stride, bias=False)

    def forward(self, w: torch.Tensor, length: int) -> torch.Tensor:
        """(B, N, F) -> (B, length), undoing :meth:`WaveEncoder.pad`."""
        y = self.deconv(w).squeeze(1)
        return y[..., self.edge:self.edge + length]
