"""Model zoo."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .tdspeakerbeam import TDSpeakerBeam
from .ueconv_tse import UEConvTSE


def build_model(model_cfg: dict[str, Any], n_speakers: int, sample_rate: int = 8000) -> nn.Module:
    """Instantiate the model selected by ``model_cfg['type']``."""
    if model_cfg["type"] == "ueconv_tse":
        return UEConvTSE(model_cfg, n_speakers, sample_rate)
    if model_cfg["type"] == "tdspeakerbeam":
        return TDSpeakerBeam(model_cfg, n_speakers, sample_rate)
    raise ValueError(f"Unknown model type {model_cfg['type']}")


__all__ = ["build_model", "UEConvTSE", "TDSpeakerBeam"]
