"""Model-agnostic inference interface used by the demos and the real-time server.

A :class:`TSEBackend` wraps one trained model. Everything outside this module talks only
to that interface (``enroll`` once, then ``extract`` whole clips or ``new_stream`` for live
audio), so a new model family plugs in by adding a backend class and registering it with
:func:`register_backend`. No caller changes are needed.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..utils import count_parameters, load_model_from_checkpoint

_BACKENDS: dict[str, type["TSEBackend"]] = {}


def register_backend(name: str) -> Callable[[type["TSEBackend"]], type["TSEBackend"]]:
    """Class decorator: make a backend available under ``backend: <name>`` in models.yaml."""
    def deco(cls: type[TSEBackend]) -> type[TSEBackend]:
        _BACKENDS[name] = cls
        return cls
    return deco


def backend_class(name: str) -> type["TSEBackend"]:
    """The backend class registered as ``name``."""
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]


@dataclass
class SpeakerProfile:
    """What a backend needs to know about the target speaker (from :meth:`TSEBackend.enroll`).

    ``data`` is backend-specific (e.g. an embedding tensor). A profile only works with the
    model that made it; ``model_id`` is checked.
    """

    model_id: str
    data: Any
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class TSEBackend(ABC):
    """One target-speaker-extraction model.

    Subclasses implement :meth:`enroll` and :meth:`extract`. Streaming defaults to the
    generic sliding-window :class:`tse.serving.streaming.ChunkedStream` built on
    :meth:`extract_window`; natively causal models can override :meth:`new_stream`.
    """

    def __init__(self, model_id: str, label: str, device: torch.device) -> None:
        self.model_id = model_id
        self.label = label
        self.device = device

    sample_rate: int = 8000

    @abstractmethod
    def enroll(self, reference: np.ndarray) -> SpeakerProfile:
        """Build the target-speaker profile from a reference recording at ``sample_rate``."""

    @abstractmethod
    def extract(self, mixture: np.ndarray, profile: SpeakerProfile) -> np.ndarray:
        """Extract the profiled speaker from a whole ``mixture`` (same length out)."""

    def extract_window(self, window: np.ndarray, profile: SpeakerProfile) -> np.ndarray:
        """Extraction of one streaming window; defaults to :meth:`extract`."""
        return self.extract(window, profile)

    def new_stream(self, profile: SpeakerProfile, **settings: Any) -> "Any":
        """A stateful live stream: ``stream.push(chunk) -> output samples``."""
        from .streaming import ChunkedStream
        return ChunkedStream(self, profile, **settings)

    def info(self) -> dict[str, Any]:
        """Static description shown in UIs."""
        return {"id": self.model_id, "label": self.label, "sample_rate": self.sample_rate,
                "device": str(self.device)}

    def _check(self, profile: SpeakerProfile) -> None:
        if profile.model_id != self.model_id:
            raise ValueError(f"speaker profile was made by model {profile.model_id!r}, "
                             f"not {self.model_id!r}; enroll again")


@register_backend("ueconv")
class UEConvBackend(TSEBackend):
    """Checkpoints of this project (U-eConv-TSE of any depth, TD-SpeakerBeam), loaded with
    :func:`tse.utils.load_model_from_checkpoint`. The enrollment is center-cropped to the
    checkpoint's ``enroll_seconds_eval`` as in evaluation."""

    def __init__(self, model_id: str, label: str, device: torch.device, checkpoint: str | Path,
                 **_: Any) -> None:
        super().__init__(model_id, label, device)
        self.checkpoint = str(checkpoint)
        self.model, ckpt = load_model_from_checkpoint(self.checkpoint, device)
        d = ckpt["config"]["data"]
        self.sample_rate = int(d["sample_rate"])
        self.enroll_samples = int(d["enroll_seconds_eval"] * self.sample_rate)
        self.n_params = count_parameters(self.model) - count_parameters(self.model.classifier)
        self.model_type = ckpt["config"]["model"]["type"]
        self.depth = ckpt["config"]["model"].get("R")
        self._lock = threading.Lock()  # one GPU call at a time per model

    def _tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None].to(self.device)

    @torch.inference_mode()
    def enroll(self, reference: np.ndarray) -> SpeakerProfile:
        ref = np.asarray(reference, dtype=np.float32)
        if len(ref) < self.sample_rate // 2:
            raise ValueError("the reference must be at least 0.5 s long")
        if len(ref) > self.enroll_samples:
            s = (len(ref) - self.enroll_samples) // 2
            ref = ref[s:s + self.enroll_samples]
        with self._lock:
            emb = self.model.embed(self._tensor(ref))
        return SpeakerProfile(self.model_id, emb, len(reference) / self.sample_rate)

    @torch.inference_mode()
    def extract(self, mixture: np.ndarray, profile: SpeakerProfile) -> np.ndarray:
        self._check(profile)
        with self._lock:
            out = self.model(self._tensor(mixture), None, compute_interference=False,
                             embedding=profile.data)["est"]
        return out[0].float().cpu().numpy()

    def info(self) -> dict[str, Any]:
        return {**super().info(), "checkpoint": self.checkpoint, "type": self.model_type,
                "R": self.depth, "params_M": round(self.n_params / 1e6, 2)}
