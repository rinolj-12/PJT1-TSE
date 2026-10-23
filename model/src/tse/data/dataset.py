"""Torch dataset, collate function and the high-level split preparation helper."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .enrollment import (Pool, build_samples, build_utterance_pool, enrollment_candidates,
                         load_or_create_enrollment_table)
from .index import dataset_tag, load_index, load_split_indexes

log = logging.getLogger(__name__)


def read_audio(path: str | Path, start: int = 0, frames: int = -1) -> np.ndarray:
    """Read a mono float32 segment ``[start, start + frames)`` with soundfile."""
    x, _ = sf.read(str(path), start=start, frames=frames, dtype="float32", always_2d=True)
    return x[:, 0]


def fit_length_tile(x: np.ndarray, n: int, start: int) -> np.ndarray:
    """Crop ``x`` to ``n`` samples from ``start``; if shorter than ``n``, tile (repeat) it."""
    if len(x) >= n:
        return x[start:start + n]
    reps = int(np.ceil(n / max(len(x), 1)))
    return np.tile(x, reps)[:n]


def fit_length_pad(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad ``x`` at the end to ``n`` samples (or crop)."""
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


class TSEDataset(Dataset):
    """Target-speaker-extraction samples.

    Each item is a dict with ``mixture (T)``, ``enrollment (Te)``, ``target (T)``,
    ``interference (T) = mixture - target``, ``speaker_label`` (int, -1 if the speaker
    is not a training speaker) and ``meta``.

    Args:
        samples: Sample table (see :func:`tse.data.enrollment.build_samples`). In eval
            mode it must also have ``enroll_path``/``enroll_length`` columns.
        root: Dataset root that the relative paths are resolved against.
        pool: Speaker utterance pool (train mode only, for re-sampling enrollments).
        spk2label: Training speaker ID -> class index.
        train: Train mode (random crops, random enrollment) vs eval mode (full length,
            fixed enrollment).
        sample_rate: Audio sample rate.
        segment_seconds: Training crop length.
        enroll_seconds: Enrollment length (train: random crop; eval: center crop).
        load_interferers: Eval mode only: also return ``interferers`` (K, T), each
            interfering source on its own (for SIR/SAR), and ``n_interferers``.
    """

    def __init__(self, samples: pd.DataFrame, root: str | Path, pool: Pool | None,
                 spk2label: dict[str, int], train: bool, sample_rate: int = 8000,
                 segment_seconds: float = 3.0, enroll_seconds: float = 4.0,
                 load_interferers: bool = False) -> None:
        self.samples = samples.reset_index(drop=True)
        self.root = Path(root)
        self.pool = pool
        self.spk2label = spk2label
        self.train = train
        self.sr = sample_rate
        self.segment = int(round(segment_seconds * sample_rate))
        self.enroll_len = int(round(enroll_seconds * sample_rate))
        self.load_interferers = load_interferers and not train
        if self.load_interferers and "interferer_paths" not in self.samples.columns:
            raise ValueError("load_interferers needs an 'interferer_paths' sample column")
        if train and pool is None:
            raise ValueError("Training mode needs an utterance pool to sample enrollments")
        if not train and "enroll_path" not in self.samples.columns:
            raise ValueError("Eval mode needs a fixed enrollment table")
        self._records = self.samples.to_dict("records")

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _randint(high: int) -> int:
        """Uniform int in ``[0, high)`` from the torch RNG (seeded per DataLoader worker)."""
        return int(torch.randint(high, (1,)).item()) if high > 0 else 0

    def _load_enrollment(self, path: str, length: int) -> np.ndarray:
        n = self.enroll_len
        if length >= n:
            start = self._randint(length - n + 1) if self.train else (length - n) // 2
            return read_audio(self.root / path, start, n)
        return fit_length_tile(read_audio(self.root / path), n, 0)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self._records[i]
        length = int(r["length"])
        if self.train:
            seg = self.segment
            start = self._randint(length - seg + 1) if length > seg else 0
            mix = fit_length_pad(read_audio(self.root / r["mix_path"], start, seg), seg)
            tgt = fit_length_pad(read_audio(self.root / r["target_path"], start, seg), seg)
            cands = enrollment_candidates(self.pool, r["speaker_id"], r["utt_id"])
            e_utt, e_path, e_len = cands[self._randint(len(cands))]
        else:
            mix = read_audio(self.root / r["mix_path"])
            tgt = read_audio(self.root / r["target_path"])
            n = min(len(mix), len(tgt))
            mix, tgt = mix[:n], tgt[:n]
            e_utt, e_path, e_len = r["enroll_utt"], r["enroll_path"], int(r["enroll_length"])
        enr = self._load_enrollment(e_path, e_len)
        meta = {
            "mixture_id": r["mixture_id"], "role": r["role"], "speaker_id": r["speaker_id"],
            "utt_id": r["utt_id"], "enroll_utt": e_utt, "enroll_path": e_path,
            "target_path": r["target_path"], "length": len(mix),
            "same_gender": r.get("same_gender", ""),
            "n_src": int(r.get("n_src", 2)),
        }
        mix_t = torch.from_numpy(np.ascontiguousarray(mix))
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt))
        item = {
            "mixture": mix_t,
            "enrollment": torch.from_numpy(np.ascontiguousarray(enr)),
            "target": tgt_t,
            "interference": mix_t - tgt_t,
            "speaker_label": self.spk2label.get(r["speaker_id"], -1),
            "meta": meta,
        }
        if self.load_interferers:
            paths = [q for q in str(r["interferer_paths"]).split("|") if q]
            srcs = [fit_length_pad(read_audio(self.root / q)[:len(mix)], len(mix)) for q in paths]
            item["interferers"] = torch.from_numpy(np.stack(srcs).astype(np.float32))
            item["n_interferers"] = len(srcs)
        return item


def collate_tse(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack a list of items, zero-padding variable-length signals to the batch maximum.

    Also returns ``lengths`` (original mixture lengths). Evaluation uses batch size 1,
    so padding only matters for custom use.
    """
    def pad_stack(key: str) -> torch.Tensor:
        xs = [b[key] for b in batch]
        n = max(x.shape[-1] for x in xs)
        return torch.stack([torch.nn.functional.pad(x, (0, n - x.shape[-1])) for x in xs])

    out = {
        "mixture": pad_stack("mixture"),
        "enrollment": pad_stack("enrollment"),
        "target": pad_stack("target"),
        "interference": pad_stack("interference"),
        "speaker_label": torch.tensor([b["speaker_label"] for b in batch], dtype=torch.long),
        "lengths": torch.tensor([b["mixture"].shape[-1] for b in batch], dtype=torch.long),
        "meta": [b["meta"] for b in batch],
    }
    if "interferers" in batch[0]:  # dynamic mixing: individual interferers (B, K, T)
        out["interferers"] = pad_stack("interferers")
        out["n_interferers"] = torch.tensor([b["n_interferers"] for b in batch], dtype=torch.long)
    return out


@dataclass
class SplitData:
    """Everything needed to build datasets for train/val/test."""

    indexes: dict[str, pd.DataFrame]
    pools: dict[str, Pool]
    samples: dict[str, pd.DataFrame]  # train: plain samples; val/test: with fixed enrollments
    dropped: dict[str, int]
    spk2label: dict[str, int]


def prepare_splits(data_cfg: dict[str, Any], rebuild: bool = False) -> SplitData:
    """Build (or load from cache) indexes, pools, samples and fixed eval enrollments."""
    indexes = load_split_indexes(data_cfg, rebuild=rebuild)
    cache_dir = Path(data_cfg["cache_dir"])
    pools, samples, dropped = {}, {}, {}
    for k, idx in indexes.items():
        pools[k] = build_utterance_pool(idx)
        s, d = build_samples(idx, pools[k])
        dropped[k] = d
        if k == "train":
            samples[k] = s
        else:
            split_name = data_cfg["val_split"] if k == "val" else data_cfg["test_split"]
            samples[k] = load_or_create_enrollment_table(
                s, pools[k], seed=data_cfg["enroll_seed"],
                path=enrollment_table_path(data_cfg, data_cfg["root"], split_name))
    spk_file = cache_dir / "train_speakers.json"
    train_spk = sorted(pools["train"].keys())
    spk2label = {s: i for i, s in enumerate(train_spk)}
    spk_file.write_text(json.dumps(spk2label, indent=0))
    return SplitData(indexes, pools, samples, dropped, spk2label)


def enrollment_table_path(data_cfg: dict[str, Any], root: str | Path, split: str) -> Path:
    """Cache path of the fixed enrollment table of ``split`` of the dataset at ``root``."""
    tag = f"{data_cfg['sample_rate']}_{data_cfg['mode']}_{data_cfg['mix_type']}"
    return Path(data_cfg["cache_dir"]) / f"enrollment_{dataset_tag(root)}{split}_{tag}.csv"


def prepare_eval_set(data_cfg: dict[str, Any], root: str | Path, split: str,
                     rebuild: bool = False) -> pd.DataFrame:
    """Samples with fixed (seeded, cached) enrollments for one split of any LibriMix-style
    dataset (e.g. Libri2Mix, Libri3Mix or our Libri4Mix-like set). The enrollment pool is
    that split's own clean sources."""
    idx = load_index(root, split, data_cfg, rebuild)
    pool = build_utterance_pool(idx)
    samples, dropped = build_samples(idx, pool)
    if dropped:
        log.warning("%s/%s: dropped %d samples without another utterance", root, split, dropped)
    return load_or_create_enrollment_table(samples, pool, seed=data_cfg["enroll_seed"],
                                           path=enrollment_table_path(data_cfg, root, split))


def val_subset(samples: pd.DataFrame, n: int | None, seed: int = 0) -> pd.DataFrame:
    """Fixed, seeded subset of ``n`` validation samples (all if ``n`` is None)."""
    if n is None or n >= len(samples):
        return samples
    idx = np.random.default_rng(seed).choice(len(samples), size=n, replace=False)
    return samples.iloc[np.sort(idx)].reset_index(drop=True)
