"""U-eConv-TSE: U-ConvBlock separator with FiLM speaker conditioning, GES and MGFA."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .ges_mgfa import GES, MGFA
from .layers import GlobalLayerNorm, WaveDecoder, WaveEncoder
from .speaker_encoder import build_speaker_encoder
from .ublock import TCNStage, UConvBlock


class TSEBase(nn.Module):
    """Shared encoder / decoder / speaker encoder / classifier for both TSE models."""

    def __init__(self, cfg: dict[str, Any], n_speakers: int, sample_rate: int = 8000) -> None:
        super().__init__()
        self.cfg = dict(cfg)
        N, L, S = cfg["N"], cfg["L"], cfg["stride"]
        self.encoder = WaveEncoder(N, L, S)
        self.decoder = WaveDecoder(N, L, S)
        self.spk_encoder = build_speaker_encoder(cfg, self.encoder, sample_rate)
        self.classifier = nn.Linear(cfg["spk_dim"], max(n_speakers, 1))
        self.mask_act = nn.ReLU() if cfg["mask_act"] == "relu" else nn.Sigmoid()

    def embed(self, enrollment: torch.Tensor) -> torch.Tensor:
        """Speaker embedding (B, spk_dim) from an enrollment waveform (B, Te)."""
        return self.spk_encoder(enrollment)


class UEConvTSE(TSEBase):
    """Proposed model.

    ``forward(mixture (B,T), enrollment (B,Te))`` returns a dict with ``est`` (B, T),
    ``interference`` (B, T) or None, ``spk_logits`` (training only, else None) and
    ``embedding``.
    """

    def __init__(self, cfg: dict[str, Any], n_speakers: int, sample_rate: int = 8000) -> None:
        super().__init__(cfg, n_speakers, sample_rate)
        N, B, H, R, D = cfg["N"], cfg["B"], cfg["H"], cfg["R"], cfg["spk_dim"]
        self.R = R
        self.inject = cfg["inject"]
        self.use_interf = cfg["use_ges_interference"]
        self.use_mgfa = cfg["use_mgfa"]
        self.bottleneck = nn.Sequential(GlobalLayerNorm(N), nn.Conv1d(N, B, 1))
        if cfg["separator"] == "ublock":
            self.stages = nn.ModuleList([
                UConvBlock(B, H, cfg["Q"], cfg["K"], D, cfg["fusion"], cfg["upsample"],
                           cfg["se_reduction"], cfg.get("attention", "se"),
                           cfg.get("attn_dim", 256), cfg.get("attn_heads", 4), depth=r)
                for r in range(R)])
        else:
            self.stages = nn.ModuleList([
                TCNStage(B, H, cfg["P"], cfg["X"], D, cfg["fusion"]) for _ in range(R)])
        self.ges = nn.ModuleList([GES(B, D, self.use_interf) for _ in range(R)])
        if self.use_mgfa:
            self.mgfa_t = nn.ModuleList([MGFA(B, cfg["mgfa_beta"]) for _ in range(R - 1)])
            if self.use_interf:
                self.mgfa_i = nn.ModuleList([MGFA(B, cfg["mgfa_beta"]) for _ in range(R - 1)])
        self.mask_t = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N, 1))
        if self.use_interf:
            self.mask_i = nn.Sequential(nn.PReLU(), nn.Conv1d(B, N, 1))

    def _aggregate(self, E: torch.Tensor | None, F_r: torch.Tensor, r: int,
                   mods: nn.ModuleList | None) -> torch.Tensor:
        if E is None or mods is None:
            return F_r
        return mods[r - 1](E, F_r)

    def forward(self, mixture: torch.Tensor, enrollment: torch.Tensor | None,
                compute_interference: bool | None = None,
                embedding: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        """``embedding`` (B, spk_dim), if given, is used instead of ``embed(enrollment)`` (e.g.
        a speaker embedding computed once for streaming); ``enrollment`` may then be None."""
        if compute_interference is None:
            compute_interference = self.training
        compute_interference = compute_interference and self.use_interf

        x_pad, T = self.encoder.pad(mixture)
        W = self.encoder(x_pad)
        e = embedding if embedding is not None else self.embed(enrollment)
        x = self.bottleneck(W)
        mgfa_t = self.mgfa_t if self.use_mgfa else None
        mgfa_i = self.mgfa_i if (self.use_mgfa and self.use_interf) else None
        E_t = E_i = None
        for r in range(self.R):
            cond = e if (self.inject == "every_stage" or r == 0) else None
            x = self.stages[r](x, cond)
            F_t, F_i = self.ges[r](x, cond, need_interference=compute_interference)
            E_t = self._aggregate(E_t, F_t, r, mgfa_t)
            if compute_interference:
                E_i = self._aggregate(E_i, F_i, r, mgfa_i)

        est = self.decoder(self.mask_act(self.mask_t(E_t)) * W, T)
        interf = None
        if compute_interference:
            interf = self.decoder(self.mask_act(self.mask_i(E_i)) * W, T)
        return {
            "est": est,
            "interference": interf,
            "spk_logits": self.classifier(e) if self.training else None,
            "embedding": e,
        }
