"""SI-SDR and the TSE training objective."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB, per signal, computed in float32.

    Args:
        est, ref: (..., T) tensors.
    Returns:
        (...,) SI-SDR values.
    """
    est = est.float()
    ref = ref.float()
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def judge_loss(judge: nn.Module, est: torch.Tensor, target: torch.Tensor,
               interferers: torch.Tensor, n_interferers: torch.Tensor,
               scale: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Speaker "judge" loss: a frozen speaker model listens to the output.

    Cosine similarities between the ECAPA embedding of ``est`` and those of the clean
    ``target`` and of each interferer, times ``scale``, are the logits of a softmax whose
    correct class is the target (an InfoNCE-style contrastive loss). The loss is low only
    when the output sounds like the target *and* unlike every interferer. Gradients flow
    into ``est``; the reference embeddings are constants.

    Args:
        judge: :class:`tse.models.speaker_encoder.FrozenECAPA`.
        est, target: (B, T). interferers: (B, K, T), zero rows after ``n_interferers``.
    Returns:
        (mean loss, fraction of items whose output is closer to the target than to every
        interferer).
    """
    B, K, T = interferers.shape
    e_out = F.normalize(judge.ecapa_embedding(est, grad=True), dim=-1)
    with torch.no_grad():
        e_tgt = F.normalize(judge.ecapa_embedding(target), dim=-1)
        e_int = F.normalize(judge.ecapa_embedding(interferers.reshape(B * K, T)), dim=-1)
        e_int = e_int.reshape(B, K, -1)
    cos_t = (e_out * e_tgt).sum(-1, keepdim=True)                 # (B, 1)
    cos_i = torch.einsum("bd,bkd->bk", e_out, e_int)              # (B, K)
    valid = torch.arange(K, device=est.device)[None] < n_interferers[:, None]
    cos_i = cos_i.masked_fill(~valid, float("-inf"))
    logits = scale * torch.cat([cos_t, cos_i], dim=1)
    loss = F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=est.device))
    correct = (cos_t[:, 0] > cos_i.max(dim=1).values).float().mean()
    return loss, correct


class TSELoss(nn.Module):
    """``-SI-SDR(est, target) + lambda_ce * CE(logits, label) + lambda_int * -SI-SDR(interf, mix - target)
    + lambda_judge * judge_loss``.

    Terms whose inputs are missing (no logits / no interference estimate / no individual
    interferers) or whose weight is zero are skipped. Labels equal to -1 are ignored by the
    CE term. The judge term needs a frozen ECAPA (``judge``) that is not part of the model,
    so checkpoints and inference are unchanged.
    """

    def __init__(self, lambda_ce: float = 0.5, lambda_int: float = 0.1,
                 lambda_judge: float = 0.0, judge: nn.Module | None = None,
                 judge_scale: float = 10.0) -> None:
        super().__init__()
        if lambda_judge > 0 and judge is None:
            raise ValueError("lambda_judge > 0 needs a judge model")
        self.lambda_ce = lambda_ce
        self.lambda_int = lambda_int
        self.lambda_judge = lambda_judge
        self.judge = judge
        self.judge_scale = judge_scale

    def forward(self, out: dict[str, Any], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Return a dict with ``total``, ``si_sdr`` (mean, dB), ``ce`` and ``int_si_sdr``."""
        sdr = si_sdr(out["est"], batch["target"]).mean()
        total = -sdr
        zero = torch.zeros((), device=sdr.device)
        ce = zero
        logits = out.get("spk_logits")
        labels = batch["speaker_label"]
        if self.lambda_ce > 0 and logits is not None and bool((labels >= 0).any()):
            ce = F.cross_entropy(logits.float(), labels, ignore_index=-1)
            total = total + self.lambda_ce * ce
        int_sdr = zero
        if self.lambda_int > 0 and out.get("interference") is not None:
            int_sdr = si_sdr(out["interference"], batch["interference"]).mean()
            total = total - self.lambda_int * int_sdr
        res = {"si_sdr": sdr.detach(), "ce": ce.detach(), "int_si_sdr": int_sdr.detach()}
        if (self.lambda_judge > 0 and "interferers" in batch
                and bool((batch["n_interferers"] > 0).any())):
            keep = batch["n_interferers"] > 0
            j, acc = judge_loss(self.judge, out["est"][keep], batch["target"][keep],
                                batch["interferers"][keep], batch["n_interferers"][keep],
                                self.judge_scale)
            total = total + self.lambda_judge * j
            res.update(judge=j.detach(), judge_acc=acc)
        res["total"] = total
        return res
