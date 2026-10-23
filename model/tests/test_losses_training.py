"""SI-SDR, loss and single training-step tests."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from conftest import small_model_cfg
from tse.engine import train_step
from tse.losses import TSELoss, si_sdr
from tse.models import build_model


def test_si_sdr_identical_is_large() -> None:
    x = torch.randn(3, 8000)
    assert (si_sdr(x, x) > 100).all()


def test_si_sdr_scale_invariant() -> None:
    torch.manual_seed(0)
    ref = torch.randn(4, 8000)
    est = ref + 0.3 * torch.randn(4, 8000)
    for s in (0.01, 3.0, 250.0):
        assert torch.allclose(si_sdr(s * est, ref), si_sdr(est, ref), atol=1e-3)


def test_si_sdr_matches_reference_formula() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        ref = rng.standard_normal(4000)
        est = ref + rng.uniform(0.1, 2) * rng.standard_normal(4000)
        r, e = ref - ref.mean(), est - est.mean()
        s_t = (e @ r) / (r @ r) * r
        expected = 10 * np.log10((s_t @ s_t) / ((e - s_t) @ (e - s_t)))
        got = float(si_sdr(torch.from_numpy(est), torch.from_numpy(ref)))
        assert abs(got - expected) < 1e-3


def test_loss_terms_and_ignore_index() -> None:
    loss = TSELoss(lambda_ce=0.5, lambda_int=0.1)
    tgt = torch.randn(2, 4000)
    out = {"est": tgt + 0.1 * torch.randn(2, 4000), "interference": None,
           "spk_logits": torch.randn(2, 3)}
    batch = {"target": tgt, "interference": torch.randn(2, 4000),
             "speaker_label": torch.tensor([-1, -1])}
    res = loss(out, batch)
    assert res["ce"] == 0 and torch.isclose(res["total"], -res["si_sdr"])


@pytest.mark.parametrize("model_type", ["ueconv_tse", "tdspeakerbeam"])
@pytest.mark.parametrize("amp", ["none", "bf16"])
def test_training_step_updates_weights(model_type: str, amp: str) -> None:
    # bf16 backward is not supported by oneDNN on every CPU; autocast is a GPU feature here
    if amp == "bf16" and not torch.cuda.is_available():
        pytest.skip("bf16 autocast test needs CUDA")
    device = torch.device("cuda" if amp == "bf16" else "cpu")
    torch.manual_seed(0)
    model = build_model(small_model_cfg(model_type), n_speakers=4).to(device)
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tgt = torch.randn(2, 8000)
    itf = torch.randn(2, 8000)
    batch = {"mixture": tgt + itf, "enrollment": torch.randn(2, 8000), "target": tgt,
             "interference": itf, "speaker_label": torch.tensor([0, 3])}
    res = train_step(model, batch, TSELoss(0.5, 0.1 if model_type == "ueconv_tse" else 0.0),
                     opt, device, amp=amp, grad_clip=5.0)
    assert res is not None and np.isfinite(res["total"])
    changed = [k for k, v in model.state_dict().items()
               if v.dtype.is_floating_point and not torch.equal(v, before[k])]
    assert len(changed) > 0
    assert any(k.startswith("spk_encoder") for k in changed)


def test_nonfinite_loss_skips_step() -> None:
    model = build_model(small_model_cfg(), n_speakers=4)
    before = copy.deepcopy(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    nan = torch.full((1, 8000), float("nan"))
    batch = {"mixture": nan, "enrollment": torch.randn(1, 8000), "target": torch.randn(1, 8000),
             "interference": torch.randn(1, 8000), "speaker_label": torch.tensor([0])}
    assert train_step(model, batch, TSELoss(), opt, torch.device("cpu")) is None
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point and "running" not in k:
            assert torch.equal(v, before[k]), k
