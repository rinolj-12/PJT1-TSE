"""Bottleneck attention variants (SE / multi-head / differential) and the SNR/SIR/SAR metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from conftest import small_model_cfg
from tse.config import DEFAULTS
from tse.metrics import snr_np, utterance_metrics
from tse.models import build_model
from tse.models.layers import (ChannelAttention, DifferentialAttention, MultiHeadAttention,
                               make_attention)
from tse.utils import load_model_from_checkpoint, save_checkpoint


@pytest.mark.parametrize("kind", ["mhsa", "diff"])
def test_attention_is_identity_at_init_and_learns(kind: str) -> None:
    torch.manual_seed(0)
    att = make_attention(kind, 32, attn_dim=16, heads=4, depth=1)
    x = torch.randn(2, 32, 21)
    assert torch.equal(att(x), x)  # zero-initialised output projection
    torch.nn.init.normal_(att.out.weight, std=0.1)
    y = att(x)
    assert y.shape == x.shape and not torch.allclose(y, x)
    y.pow(2).mean().backward()
    assert att.qkv.weight.grad is not None and att.qkv.weight.grad.abs().sum() > 0


def test_mhsa_matches_torch_multihead_attention() -> None:
    torch.manual_seed(0)
    C, D, h = 24, 16, 4
    att = MultiHeadAttention(C, D, h)
    torch.nn.init.normal_(att.out.weight, std=0.2)
    ref = torch.nn.MultiheadAttention(D, h, batch_first=True, kdim=D, vdim=D)
    x = torch.randn(2, C, 13)
    t = att.norm(x.transpose(1, 2))
    q, k, v = att.qkv(t).chunk(3, dim=-1)
    with torch.no_grad():  # identity in/out projections -> pure attention core
        ref.in_proj_weight.copy_(torch.eye(D).repeat(3, 1))
        ref.in_proj_bias.zero_()
        ref.out_proj.weight.copy_(torch.eye(D))
        ref.out_proj.bias.zero_()
    core, _ = ref(q, k, v, need_weights=False)
    expected = x + att.out(core).transpose(1, 2)
    assert torch.allclose(att(x), expected, atol=1e-5)


def test_differential_attention_matches_manual_formula() -> None:
    torch.manual_seed(0)
    C, D, heads, depth = 16, 16, 4, 2
    att = DifferentialAttention(C, D, heads, depth)
    torch.nn.init.normal_(att.out.weight, std=0.2)
    x = torch.randn(1, C, 9)
    t = att.norm(x.transpose(1, 2))[0]                     # (T, C)
    q, k, v = att.qkv(t).chunk(3, dim=-1)
    nh = heads // 2
    dh = D // nh // 2
    lam_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
    lam = (torch.exp((att.lq1 * att.lk1).sum()) - torch.exp((att.lq2 * att.lk2).sum()) + lam_init)
    outs = []
    for i in range(nh):
        qi, ki, vi = (z[:, i * 2 * dh:(i + 1) * 2 * dh] for z in (q, k, v))
        a1 = torch.softmax(qi[:, :dh] @ ki[:, :dh].T / math.sqrt(dh), -1)
        a2 = torch.softmax(qi[:, dh:] @ ki[:, dh:].T / math.sqrt(dh), -1)
        o = (a1 - lam * a2) @ vi
        o = o / torch.sqrt(o.pow(2).mean(-1, keepdim=True) + 1e-5)
        outs.append(o * (1 - lam_init))
    expected = x + att.out(torch.cat(outs, -1)).T[None]
    assert torch.allclose(att(x), expected, atol=1e-5)


def test_mhsa_and_diff_have_equal_parameter_counts() -> None:
    n = [sum(p.numel() for p in make_attention(k, 512, attn_dim=256, heads=4).parameters())
         for k in ("mhsa", "diff")]
    assert n[1] - n[0] == 4 * 64  # only the four lambda vectors (width 256 / 2 heads / 2) differ
    assert isinstance(make_attention("se", 32), ChannelAttention)
    with pytest.raises(ValueError):
        make_attention("nope", 32)


@pytest.mark.parametrize("kind", ["mhsa", "diff"])
def test_model_with_attention_runs_and_round_trips(tmp_path, kind: str) -> None:
    torch.manual_seed(0)
    cfg = small_model_cfg(attention=kind, attn_dim=16, attn_heads=4)
    model = build_model(cfg, n_speakers=5).train()
    out = model(torch.randn(2, 8013), torch.randn(2, 8000))
    assert out["est"].shape == (2, 8013) and torch.isfinite(out["est"]).all()
    model.eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, {"model": cfg, "data": dict(DEFAULTS["data"])},
                    {str(i): i for i in range(5)})
    loaded, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    mix, enr = torch.randn(1, 8000), torch.randn(1, 8000)
    with torch.no_grad():
        assert torch.allclose(model(mix, enr)["est"], loaded(mix, enr)["est"], atol=1e-6)


def test_config_without_attention_key_builds_se_model() -> None:
    cfg = small_model_cfg()
    cfg.pop("attention", None)  # checkpoints saved before the attention switch
    model = build_model(cfg, n_speakers=5)
    assert all(isinstance(s.attn, ChannelAttention) for s in model.stages)


def test_snr_and_sir_metrics() -> None:
    rng = np.random.default_rng(0)
    T = 8000
    ref, i1, i2 = (rng.standard_normal(T).astype(np.float32) for _ in range(3))
    mix = ref + 0.5 * i1 + 0.5 * i2
    assert snr_np(ref, ref) > 60
    assert abs(snr_np(0.5 * ref, ref, match_energy=False) - 20 * np.log10(2)) < 1e-6
    assert snr_np(30 * ref, ref) > 60  # energy-matched: output level does not matter
    assert snr_np(-ref, ref) < 0       # ... but the sign / waveform does
    est = ref + 0.1 * i1 + 0.01 * rng.standard_normal(T).astype(np.float32)
    m = utterance_metrics(est, ref, mix, 8000, interferers=np.stack([0.5 * i1, 0.5 * i2]))
    assert m["snri"] == pytest.approx(m["snr_out"] - m["snr_in"])
    assert m["sir_out"] == pytest.approx(20.0, abs=0.5)        # |ref|^2 / |0.1 i1|^2
    assert m["sir_in"] == pytest.approx(10 * np.log10(2), abs=0.5)  # two interferers at -6 dB
    assert m["sar_out"] == pytest.approx(40.0, abs=1.0)         # |ref|^2 / |0.01 n|^2
    assert m["siri"] == pytest.approx(m["sir_out"] - m["sir_in"])
    # without separate interferers the summed interference is used as one reference
    m1 = utterance_metrics(est, ref, mix, 8000)
    assert m1["sir_in"] == pytest.approx(m["sir_in"], abs=0.5)
