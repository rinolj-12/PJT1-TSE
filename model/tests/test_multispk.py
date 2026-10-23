"""N-speaker indexing, dynamic mixing and fine-tuning config tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tse.config import PROJECT_ROOT, load_config, resolve_paths
from tse.data.dataset import TSEDataset, prepare_eval_set, prepare_splits
from tse.data.dynamic_mix import DynamicMixDataset, active_rms


def test_three_speaker_index_and_samples(fake_libri3mix) -> None:
    d = fake_libri3mix["data_cfg"]
    samples = prepare_eval_set(d, d["root"], "test")
    assert len(samples) == 3 * fake_libri3mix["n_mixtures"]
    assert set(samples["role"]) == {"s1", "s2", "s3"}
    assert (samples["n_src"] == 3).all()
    for r in samples.itertuples(index=False):
        others = r.other_speaker_id.split("+")
        assert len(others) == 2 and r.speaker_id not in others
        assert r.enroll_utt.split("-")[0] == r.speaker_id and r.enroll_utt != r.utt_id
        # same_gender: at least one interferer shares the target's gender (odd ids = F)
        g = int(r.speaker_id) % 2
        assert r.same_gender == str(any(int(o) % 2 == g for o in others))


def test_three_speaker_dataset_items(fake_libri3mix) -> None:
    d = fake_libri3mix["data_cfg"]
    samples = prepare_eval_set(d, d["root"], "test")
    ds = TSEDataset(samples, d["root"], None, {}, train=False, enroll_seconds=2.0)
    item = ds[0]
    assert item["meta"]["n_src"] == 3
    # interference = sum of the other two sources (= mixture - target, up to 16-bit rounding)
    assert torch.allclose(item["interference"], item["mixture"] - item["target"])


def test_eval_set_enrollments_deterministic(fake_libri3mix, tmp_path) -> None:
    d = dict(fake_libri3mix["data_cfg"])
    a = prepare_eval_set(d, d["root"], "test")
    b = prepare_eval_set(dict(d, cache_dir=str(tmp_path / "other")), d["root"], "test")
    cols = ["mixture_id", "role", "enroll_utt"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def _dm_dataset(fake_librimix, **kw) -> tuple[DynamicMixDataset, dict]:
    d = fake_librimix["data_cfg"]
    data = prepare_splits(d)
    ds = DynamicMixDataset(data.samples["train"], d["root"], data.pools["train"], data.spk2label,
                           segment_seconds=0.5, enroll_seconds=1.0, **kw)
    return ds, data


def test_dynamic_mix_items(fake_librimix) -> None:
    ds, data = _dm_dataset(fake_librimix, n_spk_probs=[0.0, 1.0])  # always 3 speakers
    torch.manual_seed(0)
    for i in range(len(ds)):
        item = ds[i]
        meta = item["meta"]
        assert meta["n_src"] == 3
        assert item["mixture"].shape == (4000,) and item["enrollment"].shape == (8000,)
        assert torch.allclose(item["interference"], item["mixture"] - item["target"])
        assert float(item["mixture"].abs().max()) <= 0.95 + 1e-5
        interferer_spk = [u.split("-")[0] for u in meta["mixture_id"][3:].split("_")[1:]]
        assert meta["speaker_id"] not in interferer_spk           # target speaker never interferes
        assert len(set(interferer_spk)) == len(interferer_spk)    # distinct interferers
        assert meta["enroll_utt"] != meta["utt_id"]
        assert meta["enroll_utt"].split("-")[0] == meta["speaker_id"]
        assert item["speaker_label"] == data.spk2label[meta["speaker_id"]]


def test_dynamic_mix_speaker_count_distribution(fake_librimix) -> None:
    ds, _ = _dm_dataset(fake_librimix, n_spk_probs=[0.5, 0.3, 0.2])
    torch.manual_seed(0)
    counts = np.bincount([ds.num_speakers() for _ in range(4000)], minlength=5)[2:]
    assert np.allclose(counts / counts.sum(), [0.5, 0.3, 0.2], atol=0.03)


def test_dynamic_mix_relative_levels(fake_librimix) -> None:
    ds, _ = _dm_dataset(fake_librimix, n_spk_probs=[1.0], level_db_range=(0.0, 0.0),
                        partial_overlap_prob=0.0)
    torch.manual_seed(0)
    item = ds[0]
    t = item["target"].numpy()
    i = item["interference"].numpy()
    assert abs(20 * np.log10(active_rms(i) / active_rms(t))) < 0.1  # 0 dB when levels are fixed


def test_finetune_config_loads() -> None:
    cfg = resolve_paths(load_config(PROJECT_ROOT / "configs" / "finetune_multispk.yaml"))
    assert cfg["data"]["dynamic_mix"]["enabled"]
    assert cfg["train"]["init_from"].endswith("best.pt")
    assert [v["name"] for v in cfg["data"]["extra_val_sets"]] == ["3spk", "4spk"]
    assert all(v["root"].startswith(str(PROJECT_ROOT)) for v in cfg["data"]["extra_val_sets"])


def test_dynamic_mix_individual_interferers_and_hard_pairs(fake_librimix) -> None:
    ds0, data = _dm_dataset(fake_librimix)
    spks = sorted(data.pools["train"])
    similar = {s: [spks[(i + 1) % len(spks)]] * 3 for i, s in enumerate(spks)}  # 1 "similar" speaker
    ds, _ = _dm_dataset(fake_librimix, n_spk_probs=[1.0, 0.0, 0.0], similar_speakers=similar,
                        hard_pair_prob=1.0)
    torch.manual_seed(0)
    for i in range(len(ds)):
        item = ds[i]
        k = item["n_interferers"]
        assert item["interferers"].shape == (3, 4000)
        assert torch.allclose(item["interferers"].sum(0), item["interference"], atol=1e-5)
        assert float(item["interferers"][k:].abs().sum()) == 0.0
        if k:
            intf = item["meta"]["mixture_id"][3:].split("_")[1].split("-")[0]
            assert intf == similar[item["meta"]["speaker_id"]][0]
        assert item["meta"]["hard_pair"]
    try:
        _dm_dataset(fake_librimix, hard_pair_prob=0.5)
        raise AssertionError("hard_pair_prob without similar_speakers must fail")
    except ValueError:
        pass


class _ToyJudge(torch.nn.Module):
    """Stands in for FrozenECAPA: the embedding is a fixed projection of the waveform."""

    def __init__(self, T: int) -> None:
        super().__init__()
        self.register_buffer("W", torch.randn(T, 16, generator=torch.Generator().manual_seed(0)))

    def ecapa_embedding(self, wav: torch.Tensor, grad: bool = False) -> torch.Tensor:
        return wav @ self.W if grad else (wav @ self.W).detach()


def test_judge_loss_prefers_target_and_masks_padding() -> None:
    from tse.losses import TSELoss, judge_loss

    g = torch.Generator().manual_seed(1)
    T = 400
    tgt = torch.randn(4, T, generator=g)
    intf = torch.randn(4, 3, T, generator=g)
    n = torch.tensor([1, 2, 3, 1])
    intf[0, 1:] = 0
    intf[1, 2:] = 0
    intf[3, 1:] = 0
    judge = _ToyJudge(T)
    good, acc_good = judge_loss(judge, tgt.clone(), tgt, intf, n)
    bad, acc_bad = judge_loss(judge, intf[:, 0].clone(), tgt, intf, n)
    assert good < bad and acc_good == 1.0 and acc_bad == 0.0
    assert torch.isfinite(good) and torch.isfinite(bad)  # padded rows are masked, not NaN
    est = intf[:, 0].clone().requires_grad_(True)
    loss_fn = TSELoss(lambda_ce=0.0, lambda_int=0.0, lambda_judge=1.0, judge=judge)
    res = loss_fn({"est": est, "spk_logits": None, "interference": None},
                  {"target": tgt, "speaker_label": torch.full((4,), -1), "interferers": intf,
                   "n_interferers": n})
    res["total"].backward()
    assert est.grad is not None and "judge" in res and "judge_acc" in res
