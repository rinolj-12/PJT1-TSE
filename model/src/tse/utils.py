"""Seeding, logging, checkpoints and small helpers."""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch; optionally enable deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    """Log to stderr and, optionally, to ``log_file``."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")


def resolve_device(name: str = "auto") -> torch.device:
    """``auto`` -> cuda if available, else cpu."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of (trainable) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_checkpoint(path: str | Path, model: nn.Module, cfg: dict[str, Any],
                    spk2label: dict[str, int], optimizer: torch.optim.Optimizer | None = None,
                    scheduler: Any = None, state: dict[str, Any] | None = None) -> None:
    """Save model + config (+ optimizer/scheduler/training state) atomically."""
    ckpt = {
        "model": model.state_dict(),
        "config": cfg,
        "spk2label": spk2label,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "state": state or {},
    }
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint dict written by :func:`save_checkpoint`."""
    return torch.load(path, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model from a checkpoint's config and load its weights (eval mode)."""
    from .models import build_model

    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], len(ckpt["spk2label"]), cfg["data"]["sample_rate"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt
