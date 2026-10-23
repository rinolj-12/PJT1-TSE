"""Index / enrollment / dataset tests on a synthetic Libri2Mix tree."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from tse.data.dataset import TSEDataset, collate_tse, prepare_splits
from tse.data.enrollment import build_samples, build_utterance_pool
from tse.data.index import load_split_indexes, make_speaker_disjoint_splits, parse_mixture_id


def test_parse_mixture_id() -> None:
    assert parse_mixture_id("103-1240-0003_1235-135887-0017") == [
        ("103", "103-1240-0003"), ("1235", "1235-135887-0017")]
    assert [s for s, _ in parse_mixture_id("1-2-3_4-5-6_7-8-9_10-11-12")] == ["1", "4", "7", "10"]
    with pytest.raises(ValueError):
        parse_mixture_id("bad_id_x")


def test_each_mixture_gives_two_samples(fake_librimix) -> None:
    data = prepare_splits(fake_librimix["data_cfg"])
    for k in ("train", "val", "test"):
        idx, s = data.indexes[k], data.samples[k]
        assert data.dropped[k] == 0
        assert len(s) == 2 * len(idx)
        counts = s.groupby("mixture_id")["role"].apply(sorted)
        assert all(c == ["s1", "s2"] for c in counts)


def test_index_ignores_stale_metadata_paths(fake_librimix) -> None:
    data = prepare_splits(fake_librimix["data_cfg"])
    for p in data.indexes["train"]["mix_path"]:
        assert (fake_librimix["root"] / p).is_file()


def _check_enrollment(meta: dict, pool: dict) -> None:
    spk = meta["speaker_id"]
    assert meta["enroll_utt"].split("-")[0] == spk          # same speaker
    assert meta["enroll_utt"] != meta["utt_id"]             # different utterance
    assert meta["enroll_path"] != meta["target_path"]       # never the target file
    assert meta["enroll_utt"] in {u for u, _, _ in pool[spk]}  # from the same split


def test_train_enrollment_constraints(fake_librimix) -> None:
    d = fake_librimix["data_cfg"]
    data = prepare_splits(d)
    ds = TSEDataset(data.samples["train"], d["root"], data.pools["train"], data.spk2label,
                    train=True, segment_seconds=0.5, enroll_seconds=1.0)
    torch.manual_seed(0)
    for _ in range(5):
        for i in range(len(ds)):
            item = ds[i]
            _check_enrollment(item["meta"], data.pools["train"])
            assert item["mixture"].shape == (4000,) and item["enrollment"].shape == (8000,)
            assert torch.allclose(item["interference"], item["mixture"] - item["target"])
            assert item["speaker_label"] == data.spk2label[item["meta"]["speaker_id"]]


def test_eval_enrollment_constraints_and_tiling(fake_librimix) -> None:
    d = fake_librimix["data_cfg"]
    data = prepare_splits(d)
    ds = TSEDataset(data.samples["test"], d["root"], None, data.spk2label, train=False,
                    enroll_seconds=6.0)
    for i in range(len(ds)):
        item = ds[i]
        _check_enrollment(item["meta"], data.pools["test"])
        assert item["enrollment"].shape == (48000,)  # tiled: sources are 1-2 s long
        assert item["speaker_label"] == -1  # not a training speaker
        assert item["mixture"].shape == item["target"].shape
    batch = collate_tse([ds[0], ds[1]])
    assert batch["mixture"].shape[0] == 2 and batch["lengths"].shape == (2,)


def test_eval_enrollments_deterministic(fake_librimix, tmp_path) -> None:
    d = dict(fake_librimix["data_cfg"])
    a = prepare_splits(d).samples["test"]
    d2 = dict(d, cache_dir=str(tmp_path / "cache2"))  # fresh construction, no cached table
    b = prepare_splits(d2).samples["test"]
    cols = ["mixture_id", "role", "enroll_utt", "enroll_path"]
    pd.testing.assert_frame_equal(a[cols].reset_index(drop=True), b[cols].reset_index(drop=True))


def test_speaker_without_other_utterance_is_dropped(fake_librimix) -> None:
    idx = load_split_indexes(fake_librimix["data_cfg"])["train"]
    lonely = pd.DataFrame([{**idx.iloc[0].to_dict(), "mixture_id": "999-1-0001_100-1-0000",
                            "spk1": "999", "utt1": "999-1-0001"}])
    idx2 = pd.concat([idx, lonely], ignore_index=True)
    samples, dropped = build_samples(idx2, build_utterance_pool(idx2))
    assert dropped == 1
    assert "999" not in set(samples["speaker_id"])


def test_custom_split_is_speaker_disjoint(fake_librimix) -> None:
    idx = load_split_indexes(fake_librimix["data_cfg"])
    allidx = pd.concat(list(idx.values()), ignore_index=True)
    parts = make_speaker_disjoint_splits(allidx, [0.6, 0.2, 0.2], seed=0)
    spk = {k: set(v["spk1"]) | set(v["spk2"]) for k, v in parts.items()}
    assert not (spk["train"] & spk["val"]) and not (spk["train"] & spk["test"])
    assert not (spk["val"] & spk["test"])


def test_custom_split_used_when_eval_metadata_missing(fake_librimix) -> None:
    d = fake_librimix["data_cfg"]
    meta = fake_librimix["root"] / "wav8k" / "min" / "metadata"
    (meta / "mixture_dev_mix_clean.csv").unlink()
    parts = load_split_indexes(d)
    spk = {k: set(v["spk1"]) | set(v["spk2"]) for k, v in parts.items()}
    assert not (spk["train"] & spk["val"]) and not (spk["train"] & spk["test"])
    assert not (spk["val"] & spk["test"])
    from pathlib import Path
    assert (Path(d["splits_dir"]) / "train.csv").is_file()


def test_real_splits_speaker_disjoint(fake_librimix) -> None:
    data = prepare_splits(fake_librimix["data_cfg"])
    s = {k: set(v) for k, v in data.pools.items()}
    assert not (s["train"] & s["val"]) and not (s["train"] & s["test"])
    assert data.samples["test"]["same_gender"].isin(["True", "False"]).all()
