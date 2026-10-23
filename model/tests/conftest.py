"""Shared fixtures: small model configs and a synthetic Libri2Mix tree."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from tse.config import DEFAULTS


def small_model_cfg(model_type: str = "ueconv_tse", **over) -> dict:
    """A small but structurally complete model config (Q=4, stride=8 like the real one)."""
    cfg = copy.deepcopy(DEFAULTS["model"])
    cfg.update(type=model_type, N=32, B=16, H=32, Q=4, K=5, R=2, spk_dim=16,
               spk_channels=[16, 32, 32], X=3, adapt_layer=2)
    cfg.update(over)
    return cfg


def _write_split(root: Path, split: str, mixtures: list[tuple[str, ...]], rng: np.random.Generator,
                 sr: int = 8000) -> None:
    sdir = root / "wav8k" / "min"
    n_src = len(mixtures[0])
    for d in ["mix_clean"] + [f"s{k}" for k in range(1, n_src + 1)]:
        (sdir / split / d).mkdir(parents=True, exist_ok=True)
    rows = []
    for utts in mixtures:
        mid = "_".join(utts)
        n = int(sr * rng.uniform(1.0, 2.0))
        srcs = [(0.1 * rng.standard_normal(n)).astype(np.float32) for _ in utts]
        for k, x in enumerate(srcs, start=1):
            sf.write(sdir / split / f"s{k}" / f"{mid}.wav", x, sr)
        sf.write(sdir / split / "mix_clean" / f"{mid}.wav", np.sum(srcs, axis=0), sr)
        row = {"mixture_ID": mid, "mixture_path": "C:/elsewhere/x.wav", "length": n}
        row.update({f"source_{k}_path": "C:/elsewhere/y.wav" for k in range(1, n_src + 1)})
        rows.append(row)
    (sdir / "metadata").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(sdir / "metadata" / f"mixture_{split}_mix_clean.csv", index=False)


@pytest.fixture
def fake_librimix(tmp_path: Path) -> dict:
    """Synthetic Libri2Mix with disjoint speakers per split and a SPEAKERS.TXT."""
    rng = np.random.default_rng(0)
    root = tmp_path / "Libri2Mix"
    spk = {"train-100": ["100", "101", "102", "103"], "dev": ["200", "201", "202"],
           "test": ["300", "301", "302"]}
    for split, speakers in spk.items():
        mixtures = []
        for i in range(6):
            a, b = speakers[i % len(speakers)], speakers[(i + 1) % len(speakers)]
            mixtures.append((f"{a}-1-{i:04d}", f"{b}-2-{i:04d}"))
        # an utterance reused in two mixtures (like real dev/test)
        mixtures.append((mixtures[0][0], f"{speakers[2]}-3-0099"))
        _write_split(root, split, mixtures, rng)
    genders = {s: ("F" if int(s) % 2 else "M") for v in spk.values() for s in v}
    with open(root / "SPEAKERS.TXT", "w", encoding="utf-8") as f:
        f.write(";ID |SEX| SUBSET | MINUTES | NAME\n")
        for s, g in genders.items():
            f.write(f"{s} | {g} | x | 1.0 | n\n")
    data_cfg = copy.deepcopy(DEFAULTS["data"])
    data_cfg.update(root=str(root), cache_dir=str(tmp_path / "cache"),
                    splits_dir=str(tmp_path / "splits"))
    return {"root": root, "data_cfg": data_cfg, "speakers": spk}


@pytest.fixture
def fake_libri3mix(tmp_path: Path) -> dict:
    """Synthetic 3-speaker test set (5 speakers, SPEAKERS.TXT with genders)."""
    rng = np.random.default_rng(1)
    root = tmp_path / "Libri3Mix"
    spk = ["300", "301", "302", "303", "304"]
    mixtures = [tuple(f"{spk[(i + j) % 5]}-{j + 1}-{i:04d}" for j in range(3)) for i in range(8)]
    _write_split(root, "test", mixtures, rng)
    with open(root / "SPEAKERS.TXT", "w", encoding="utf-8") as f:
        for s in spk:
            f.write(f"{s} | {'F' if int(s) % 2 else 'M'} | x | 1.0 | n\n")
    data_cfg = copy.deepcopy(DEFAULTS["data"])
    data_cfg.update(root=str(root), cache_dir=str(tmp_path / "cache3"),
                    splits_dir=str(tmp_path / "splits3"))
    return {"root": root, "data_cfg": data_cfg, "n_mixtures": len(mixtures)}
