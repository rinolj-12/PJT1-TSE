"""TD-SpeakerBeam baseline: Conv-TasNet separator with multiplicative speaker adaptation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .layers import GlobalLayerNorm
from .ublock import TCNBlock
from .ueconv_tse import TSEBase


class TDSpeakerBeam(TSEBase):
    """Conv-TasNet (R repeats of X dilated TCN blocks, non-causal, gLN) with the output of
    TCN block ``adapt_layer`` (0-based over all R*X blocks; 7 = end of the first repeat
    for X=8) multiplied element-wise by ``Linear(e)``.

    Uses the same encoder/decoder and speaker encoder as :class:`UEConvTSE`. Has no
    interference branch.
    """

    def __init__(self, cfg: dict[str, Any], n_speakers: int, sample_rate: int = 8000) -> None:
        super().__init__(cfg, n_speakers, sample_rate)
        N, B, H, P, X, R = cfg["N"], cfg["B"], cfg["H"], cfg["P"], cfg["X"], cfg["R"]
        self.adapt_layer = cfg["adapt_layer"]
        if not 0 <= self.adapt_layer < R * X:
            raise ValueError(f"adapt_layer={self.adapt_layer} outside [0, {R * X})")
        self.bottleneck = nn.Sequential(GlobalLayerNorm(N), nn.Conv1d(N, B, 1))
        self.blocks = nn.ModuleList([TCNBlock(B, H, P, 2 ** (i % X)) for i in range(R * X)])
        self.adapt = nn.Linear(cfg["spk_dim"], B)
        self.mask = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N, 1))

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor | None,
                compute_interference: bool | None = None,
                embedding: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        """``embedding`` (B, spk_dim), if given, is used instead of ``embed(enrollment)`` (e.g.
        a speaker embedding computed once for streaming); ``enrollment`` may then be None."""
        x_pad, T = self.encoder.pad(mixture)
        W = self.encoder(x_pad)
        e = embedding if embedding is not None else self.embed(enrollment)
        x = self.bottleneck(W)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == self.adapt_layer:
                x = x * self.adapt(e).unsqueeze(-1)
        est = self.decoder(self.mask_act(self.mask(x)) * W, T)
        return {
            "est": est,
            "interference": None,
            "spk_logits": self.classifier(e) if self.training else None,
            "embedding": e,
        }
