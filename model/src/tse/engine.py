"""Training and validation loops."""

from __future__ import annotations

import contextlib
import logging
import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import TSELoss, si_sdr

log = logging.getLogger(__name__)


def autocast_ctx(device: torch.device, amp: str) -> contextlib.AbstractContextManager:
    """bf16 autocast for the model forward, or a no-op."""
    if amp == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor entries of a batch to ``device``."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def train_step(model: nn.Module, batch: dict[str, Any], loss_fn: TSELoss,
               optimizer: torch.optim.Optimizer, device: torch.device, amp: str = "none",
               grad_clip: float | None = 5.0) -> dict[str, float] | None:
    """One optimisation step. Returns loss components, or None if the loss was non-finite
    (in which case the step is skipped)."""
    model.train()
    batch = to_device(batch, device)
    with autocast_ctx(device, amp):
        out = model(batch["mixture"], batch["enrollment"],
                    compute_interference=loss_fn.lambda_int > 0)
    losses = loss_fn(out, batch)  # SI-SDR / CE in float32, outside autocast
    if not torch.isfinite(losses["total"]):
        log.warning("Non-finite loss (%s); skipping optimizer step", losses["total"].item())
        optimizer.zero_grad(set_to_none=True)
        return None
    optimizer.zero_grad(set_to_none=True)
    losses["total"].backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           grad_clip if grad_clip else float("inf"))
    if not torch.isfinite(gnorm):
        log.warning("Non-finite gradient norm; skipping optimizer step")
        optimizer.zero_grad(set_to_none=True)
        return None
    optimizer.step()
    res = {k: float(v.detach()) for k, v in losses.items()}
    res["grad_norm"] = float(gnorm)
    return res


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: TSELoss,
                    optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    train_cfg: dict[str, Any], writer: Any = None,
                    global_step: int = 0) -> tuple[dict[str, float], int]:
    """Train for one epoch (or ``max_steps_per_epoch`` steps). Returns (mean stats, step)."""
    sums: dict[str, float] = {}
    n_ok = n_skipped = 0
    max_steps = train_cfg.get("max_steps_per_epoch")
    t0 = time.time()
    for i, batch in enumerate(loader):
        if max_steps is not None and i >= max_steps:
            break
        res = train_step(model, batch, loss_fn, optimizer, device, train_cfg["amp"],
                         train_cfg["grad_clip"])
        global_step += 1
        if res is None:
            n_skipped += 1
            continue
        n_ok += 1
        for k, v in res.items():
            sums[k] = sums.get(k, 0.0) + v
        if writer is not None and global_step % 10 == 0:
            for k, v in res.items():
                writer.add_scalar(f"train_step/{k}", v, global_step)
        if (i + 1) % train_cfg["log_every"] == 0:
            judge = (f"  judge {sums['judge'] / n_ok:.3f} acc {sums['judge_acc'] / n_ok:.3f}"
                     if "judge" in sums else "")
            log.info("epoch %d step %d/%d  loss %.3f  si_sdr %.2f dB  ce %.3f%s  (%.1f it/s)",
                     epoch, i + 1, len(loader), sums["total"] / n_ok, sums["si_sdr"] / n_ok,
                     sums["ce"] / n_ok, judge, (i + 1) / (time.time() - t0))
    stats = {k: v / max(n_ok, 1) for k, v in sums.items()}
    stats["skipped_steps"] = float(n_skipped)
    stats["epoch_time_s"] = time.time() - t0
    stats["steps"] = float(n_ok + n_skipped)
    return stats, global_step


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device,
             amp: str = "none") -> dict[str, float]:
    """Full-length validation: mean SI-SDR, SI-SDRi and negative-SI-SDRi rate (NSR).

    cuDNN benchmark mode is disabled here: every utterance has a different length and
    re-autotuning per shape would dominate the run time.
    """
    model.eval()
    prev_benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        imp, out_sdr = _validate_loop(model, loader, device, amp)
    finally:
        torch.backends.cudnn.benchmark = prev_benchmark
    imp_a = np.array(imp)
    res = {"si_sdr": float(np.mean(out_sdr)), "si_sdri": float(imp_a.mean()),
           "nsr": float((imp_a < 0).mean())}
    if not all(math.isfinite(v) for v in res.values()):
        log.warning("Non-finite validation metrics: %s", res)
    return res


def _validate_loop(model: nn.Module, loader: DataLoader, device: torch.device,
                   amp: str) -> tuple[list[float], list[float]]:
    out_sdr, imp = [], []
    for batch in loader:
        batch = to_device(batch, device)
        with autocast_ctx(device, amp):
            est = model(batch["mixture"], batch["enrollment"], compute_interference=False)["est"]
        for b in range(est.shape[0]):
            n = int(batch["lengths"][b])
            s_out = float(si_sdr(est[b, :n], batch["target"][b, :n]))
            s_in = float(si_sdr(batch["mixture"][b, :n], batch["target"][b, :n]))
            out_sdr.append(s_out)
            imp.append(s_out - s_in)
    return imp, out_sdr
