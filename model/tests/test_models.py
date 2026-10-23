"""Model shape, U-block, MGFA, GES, interference-branch and checkpoint tests."""

from __future__ import annotations

import pytest
import torch

from conftest import small_model_cfg
from tse.models import build_model
from tse.models.ges_mgfa import GES, MGFA
from tse.models.layers import FiLM
from tse.models.ublock import UConvBlock
from tse.utils import load_model_from_checkpoint, save_checkpoint


@pytest.mark.parametrize("model_type", ["ueconv_tse", "tdspeakerbeam"])
@pytest.mark.parametrize("T", [24000, 24013, 31777])
def test_output_length_matches_input(model_type: str, T: int) -> None:
    torch.manual_seed(0)
    model = build_model(small_model_cfg(model_type), n_speakers=5).eval()
    mix, enr = torch.randn(2, T), torch.randn(2, 16000)
    with torch.no_grad():
        out = model(mix, enr)
    assert out["est"].shape == (2, T)
    assert torch.isfinite(out["est"]).all()


@pytest.mark.parametrize("over", [
    {"separator": "tcn"}, {"fusion": "mult"}, {"fusion": "concat"}, {"inject": "input_only"},
    {"use_ges_interference": False}, {"use_mgfa": False}, {"mgfa_beta": "scalar"},
    {"upsample": "transposed"}, {"share_encoder": False}, {"mask_act": "sigmoid"},
])
def test_ablation_variants_run(over: dict) -> None:
    model = build_model(small_model_cfg(**over), n_speakers=5).train()
    out = model(torch.randn(2, 8013), torch.randn(2, 8000))
    assert out["est"].shape == (2, 8013)
    if over.get("use_ges_interference") is False:
        assert out["interference"] is None
    else:
        assert out["interference"].shape == (2, 8013)


def test_ublock_shape_and_film_identity_at_init() -> None:
    torch.manual_seed(0)
    blk = UConvBlock(B=16, H=32, Q=4, K=5, spk_dim=8).eval()
    x = torch.randn(2, 16, 333)  # not a multiple of 2**Q
    e1, e2 = torch.randn(2, 8), torch.randn(2, 8)
    y1 = blk(x, e1)
    assert y1.shape == x.shape
    # FiLM is the identity at init: output independent of e and equal to the unconditioned block
    assert torch.allclose(y1, blk(x, e2), atol=1e-6)
    assert torch.allclose(y1, blk(x, None), atol=1e-6)


def test_film_identity_with_gamma1_beta0() -> None:
    film = FiLM(8, 4)
    x, e = torch.randn(2, 4, 10), torch.randn(2, 8)
    assert torch.allclose(film(x, e), x)


def test_mgfa_matches_manual_recursion() -> None:
    torch.manual_seed(0)
    for mode in ("scalar", "channel"):
        m = MGFA(6, mode)
        with torch.no_grad():
            m.logit.normal_()
        Fs = [torch.randn(2, 6, 9) for _ in range(4)]
        E = None
        for F_r in Fs:
            E = m(E, F_r)
        b = torch.sigmoid(m.logit)
        ref = Fs[0]
        for F_r in Fs[1:]:
            ref = b * ref + (1 - b) * F_r
        assert torch.allclose(E, ref, atol=1e-6)


def test_mgfa_first_stage_is_identity_and_beta_init() -> None:
    m = MGFA(4, "channel")
    with torch.no_grad():
        m.logit.fill_(2.0)  # beta fixed
    F1 = torch.randn(1, 4, 5)
    assert torch.equal(m(None, F1), F1)
    assert torch.allclose(MGFA(4, "scalar").beta, torch.tensor(0.5))


def test_ges_shapes_and_gate_depends_on_e() -> None:
    torch.manual_seed(0)
    ges = GES(B=8, spk_dim=4)
    x = torch.randn(2, 8, 20)
    Ft, Fi = ges(x, torch.randn(2, 4))
    assert Ft.shape == (2, 8, 20) and Fi.shape == (2, 8, 20)
    g1, g2 = ges.gate(x, torch.randn(2, 4)), ges.gate(x, torch.randn(2, 4))
    assert not torch.allclose(g1, g2)


def test_ges_target_only_path_is_exact() -> None:
    torch.manual_seed(0)
    ges = GES(B=8, spk_dim=4)
    with torch.no_grad():
        ges.norm.weight.normal_()
        ges.norm.bias.normal_()
    x, e = torch.randn(2, 8, 20), torch.randn(2, 4)
    Ft_full, _ = ges(x, e, need_interference=True)
    Ft_only, Fi = ges(x, e, need_interference=False)
    assert Fi is None
    assert torch.allclose(Ft_full, Ft_only, atol=1e-5)


def test_interference_branch_not_computed_in_inference() -> None:
    model = build_model(small_model_cfg(), n_speakers=5).eval()
    calls = {"n": 0}
    model.mask_i.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    for m in model.mgfa_i:
        m.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    with torch.no_grad():
        out = model(torch.randn(1, 8000), torch.randn(1, 8000), compute_interference=False)
        out_default = model(torch.randn(1, 8000), torch.randn(1, 8000))  # eval default
    assert out["interference"] is None and out_default["interference"] is None
    assert calls["n"] == 0
    model.train()
    out = model(torch.randn(2, 8000), torch.randn(2, 8000))
    assert out["interference"] is not None and calls["n"] > 0


@pytest.mark.parametrize("model_type", ["ueconv_tse", "tdspeakerbeam"])
def test_checkpoint_round_trip(tmp_path, model_type: str) -> None:
    torch.manual_seed(0)
    from tse.config import DEFAULTS
    cfg = {"model": small_model_cfg(model_type), "data": dict(DEFAULTS["data"])}
    model = build_model(cfg["model"], n_speakers=5)
    model.train()
    model(torch.randn(2, 8000), torch.randn(2, 8000))  # update BN running stats
    model.eval()
    mix, enr = torch.randn(1, 12345), torch.randn(1, 8000)
    with torch.no_grad():
        ref = model(mix, enr)["est"]
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, cfg, {str(i): i for i in range(5)})
    loaded, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    with torch.no_grad():
        out = loaded(mix, enr)["est"]
    assert torch.equal(ref, out)


def test_ecapa_without_speechbrain_raises_clear_error() -> None:
    pytest.importorskip("torchaudio")
    try:
        import speechbrain  # noqa: F401
        pytest.skip("speechbrain installed")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="speechbrain"):
        build_model(small_model_cfg(spk_encoder="ecapa_frozen"), n_speakers=5)


def test_ecapa_frozen_encoder_is_frozen_and_round_trips(tmp_path) -> None:
    pytest.importorskip("speechbrain")
    from tse.config import PROJECT_ROOT
    if not (PROJECT_ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb" / "embedding_model.ckpt").is_file():
        pytest.skip("pretrained ECAPA not downloaded")
    torch.manual_seed(0)
    cfg = small_model_cfg(spk_encoder="ecapa_frozen")
    model = build_model(cfg, n_speakers=5).train()
    enc = model.spk_encoder
    assert not enc.embedding_model.training and enc.proj.training
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all(not n.startswith("spk_encoder.") or n.startswith("spk_encoder.proj.") for n in trainable)
    out = model(torch.randn(2, 8013), torch.randn(2, 16000))
    (out["est"].pow(2).mean() + out["spk_logits"].mean()).backward()
    assert enc.proj.weight.grad is not None
    assert all(p.grad is None for p in enc.embedding_model.parameters())
    model.eval()
    mix, enr = torch.randn(1, 8000), torch.randn(1, 16000)
    with torch.no_grad():
        ref = model(mix, enr)["est"]
    path = tmp_path / "ecapa.pt"
    save_checkpoint(path, model, {"model": cfg, "data": {"sample_rate": 8000}}, {str(i): i for i in range(5)})
    loaded, _ = load_model_from_checkpoint(path, torch.device("cpu"))
    with torch.no_grad():
        assert torch.allclose(loaded(mix, enr)["est"], ref, atol=1e-6)
