"""Gated Embedding Split (GES) and Multi-stage Gated Feature Aggregation (MGFA)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GES(nn.Module):
    """Split stage features into target and interference streams, gated by the speaker.

    ``a, g = split(Conv1x1(x))``; ``glu = a * sigmoid(g + W e)``;
    ``F_t, F_i = split(GroupNorm(2, 2B)(Conv1x1(glu)))``.

    With ``use_interference=False`` the module only has a target head.
    Because GroupNorm with 2 groups normalises each half independently, the target half
    can be computed alone (``need_interference=False``) with identical results.
    """

    def __init__(self, B: int, spk_dim: int, use_interference: bool = True) -> None:
        super().__init__()
        self.B = B
        self.use_interference = use_interference
        self.inp = nn.Conv1d(B, 2 * B, 1)
        self.spk_gate = nn.Linear(spk_dim, B)
        n_out = 2 if use_interference else 1
        self.out = nn.Conv1d(B, n_out * B, 1)
        self.norm = nn.GroupNorm(n_out, n_out * B)

    def gate(self, x: torch.Tensor, e: torch.Tensor | None) -> torch.Tensor:
        """Return the GLU gate ``sigmoid(g + W e)`` (B, B, F)."""
        _, g = self.inp(x).chunk(2, dim=1)
        if e is not None:
            g = g + self.spk_gate(e).unsqueeze(-1)
        return torch.sigmoid(g)

    def forward(self, x: torch.Tensor, e: torch.Tensor | None,
                need_interference: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B, Bc, F) -> (F_t, F_i or None), each (B, Bc, F)."""
        a, g = self.inp(x).chunk(2, dim=1)
        if e is not None:
            g = g + self.spk_gate(e).unsqueeze(-1)
        glu = a * torch.sigmoid(g)
        Bc = self.B
        if self.use_interference and need_interference:
            y = self.norm(self.out(glu))
            return y[:, :Bc], y[:, Bc:]
        # target half only: conv with the first Bc output channels + its own group norm
        y = F.conv1d(glu, self.out.weight[:Bc], self.out.bias[:Bc])
        y = F.group_norm(y, 1, self.norm.weight[:Bc], self.norm.bias[:Bc], self.norm.eps)
        return y, None


class MGFA(nn.Module):
    """Gated running aggregation ``E_r = beta * E_{r-1} + (1 - beta) * F_r`` (``E_1 = F_1``).

    ``beta = sigmoid(logit)`` with a single logit (``scalar``) or one per channel
    (``channel``); logits start at 0, i.e. beta = 0.5.
    """

    def __init__(self, channels: int, beta_mode: str = "channel") -> None:
        super().__init__()
        n = channels if beta_mode == "channel" else 1
        self.logit = nn.Parameter(torch.zeros(1, n, 1))

    @property
    def beta(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(self, E_prev: torch.Tensor | None, F_r: torch.Tensor) -> torch.Tensor:
        if E_prev is None:
            return F_r
        b = self.beta.to(F_r.dtype)
        return b * E_prev + (1.0 - b) * F_r
