"""U-ConvBlock with speaker fusion at the coarsest scale, and a TCN stage alternative."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DSConvBlock, GlobalLayerNorm, make_attention, make_fusion


class UConvBlock(nn.Module):
    """One separator stage: U-shaped multi-resolution conv block with a residual path.

    ``x (B, Bc, F) -> 1x1 (Bc->H) -> Q stride-2 DS-conv downsamplings -> fusion(e) +
    attention at F/2^Q -> upsample + skip-sum back to F -> 1x1 (H->Bc) -> + x``.

    ``attention`` is ``se`` (squeeze-and-excitation channel attention), ``mhsa``
    (multi-head self-attention over time) or ``diff`` (differential attention over time);
    ``depth`` (stage index) sets the differential-attention ``lambda_init``.

    The time axis is zero-padded to a multiple of ``2**Q`` internally and cropped back,
    so any length works.
    """

    def __init__(self, B: int, H: int, Q: int, K: int, spk_dim: int, fusion: str = "film",
                 upsample: str = "nearest", se_reduction: int = 8, attention: str = "se",
                 attn_dim: int = 256, attn_heads: int = 4, depth: int = 0) -> None:
        super().__init__()
        self.Q = Q
        self.proj_in = nn.Sequential(nn.Conv1d(B, H, 1), nn.PReLU(), GlobalLayerNorm(H))
        self.down = nn.ModuleList([DSConvBlock(H, H, K, stride=2) for _ in range(Q)])
        self.fusion = make_fusion(fusion, spk_dim, H)
        self.attn = make_attention(attention, H, se_reduction, attn_dim, attn_heads, depth)
        self.upsample = upsample
        if upsample == "transposed":
            self.up = nn.ModuleList([nn.ConvTranspose1d(H, H, 4, stride=2, padding=1, groups=H)
                                     for _ in range(Q)])
        self.proj_out = nn.Conv1d(H, B, 1)

    def _up(self, y: torch.Tensor, q: int, size: int) -> torch.Tensor:
        if self.upsample == "transposed":
            y = self.up[q](y)
            return y[..., :size]
        return F.interpolate(y, size=size, mode="nearest")

    def forward(self, x: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        """x: (B, Bc, F); e: (B, D) or None (no conditioning in this stage)."""
        T = x.shape[-1]
        mult = 2 ** self.Q
        pad = (mult - T % mult) % mult
        y = self.proj_in(F.pad(x, (0, pad)))
        skips = [y]
        for d in self.down:
            y = d(y)
            skips.append(y)
        if e is not None:
            y = self.fusion(y, e)
        y = self.attn(y)
        for q in reversed(range(self.Q)):
            y = self._up(y, q, skips[q].shape[-1]) + skips[q]
        return self.proj_out(y)[..., :T] + x


class TCNBlock(nn.Module):
    """Conv-TasNet block: 1x1 -> PReLU -> gLN -> dilated depthwise -> PReLU -> gLN -> 1x1, residual."""

    def __init__(self, B: int, H: int, P: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(B, H, 1), nn.PReLU(), GlobalLayerNorm(H),
            nn.Conv1d(H, H, P, padding=dilation * (P - 1) // 2, dilation=dilation, groups=H),
            nn.PReLU(), GlobalLayerNorm(H),
            nn.Conv1d(H, B, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNStage(nn.Module):
    """``separator: tcn`` ablation stage: speaker fusion at the stage input, then X TCN blocks."""

    def __init__(self, B: int, H: int, P: int, X: int, spk_dim: int, fusion: str = "film") -> None:
        super().__init__()
        self.fusion = make_fusion(fusion, spk_dim, B)
        self.blocks = nn.Sequential(*[TCNBlock(B, H, P, 2 ** i) for i in range(X)])

    def forward(self, x: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        y = self.fusion(x, e) if e is not None else x
        return self.blocks(y)
