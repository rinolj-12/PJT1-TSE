"""Core of the interactive web demo (``scripts/web_demo.py``).

The user supplies 2-4 separate single-speaker recordings and a reference clip of the
target speaker. :func:`mix_sources` levels and sums them into a mixture, a model from
the registry (:mod:`tse.serving`) runs on it, and :func:`score_extraction`
compares the output with every source, which is possible because the clean sources are
known. :class:`SpeakerPool` draws random LibriSpeech test speakers (with a second
utterance as the reference) so the demo also works without any recordings.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .data.dataset import read_audio
from .data.dynamic_mix import active_rms
from .data.enrollment import build_utterance_pool
from .data.index import load_index, source_fields
from .metrics import si_sdr_np

log = logging.getLogger(__name__)

SOURCE_RMS = 0.05  # common level (about -26 dBFS) every source is scaled to before its gain


def fit_to_length(x: np.ndarray, n: int) -> np.ndarray:
    """Crop or zero-pad (at the end) ``x`` to exactly ``n`` samples."""
    x = x[:n]
    return np.pad(x, (0, n - len(x))) if len(x) < n else x


def mix_sources(sources: list[np.ndarray], gains_db: list[float], length_mode: str = "longest",
                peak: float = 0.9) -> tuple[np.ndarray, list[np.ndarray]]:
    """Level, align and sum single-speaker ``sources`` into a mixture.

    Each source is scaled to the same active RMS and then by its gain in dB, so a gain of 0
    for everyone gives equally loud speakers. ``length_mode`` is ``longest`` (shorter
    sources are zero-padded) or ``shortest`` (all are cropped, like LibriMix ``min``). If the
    sum would clip, the mixture and all sources are scaled down together.

    Returns ``(mixture, scaled_sources)``; ``sum(scaled_sources) == mixture``.
    """
    if len(sources) != len(gains_db):
        raise ValueError("need one gain per source")
    if length_mode not in ("longest", "shortest"):
        raise ValueError(f"length_mode must be 'longest' or 'shortest', got {length_mode!r}")
    lengths = [len(s) for s in sources]
    n = max(lengths) if length_mode == "longest" else min(lengths)
    if n == 0:
        raise ValueError("empty source")
    scaled = []
    for s, g in zip(sources, gains_db):
        s = fit_to_length(np.asarray(s, dtype=np.float32), n)
        r = active_rms(s)
        if r <= 0:
            raise ValueError("a source is silent")
        scaled.append(s * (SOURCE_RMS / r) * 10 ** (g / 20))
    mixture = np.sum(scaled, axis=0)
    m = float(np.abs(mixture).max())
    if m > peak:
        scaled = [s * (peak / m) for s in scaled]
        mixture = mixture * (peak / m)
    return mixture.astype(np.float32), [s.astype(np.float32) for s in scaled]


def score_extraction(est: np.ndarray, sources: list[np.ndarray], mixture: np.ndarray,
                     target: int) -> dict[str, Any]:
    """SI-SDR of the input and output against the target, plus the output against every
    source (which speaker the output actually resembles)."""
    tgt = sources[target]
    out = si_sdr_np(est, tgt)
    inp = si_sdr_np(mixture, tgt)
    per_source = [si_sdr_np(est, s) for s in sources]
    return {
        "si_sdr_in": inp,
        "si_sdr_out": out,
        "si_sdri": out - inp,
        "si_sdr_vs_sources": per_source,
        "closest_source": int(np.argmax(per_source)),
    }


class SpeakerPool:
    """Utterances of the speakers of one LibriMix split, for random demo inputs."""

    def __init__(self, data_cfg: dict[str, Any], split: str = "test") -> None:
        self.root = Path(data_cfg["root"])
        idx = load_index(self.root, split, data_cfg)
        self.pool = build_utterance_pool(idx)
        self.genders: dict[str, str] = {}
        for row in idx.itertuples(index=False):
            for k in range(1, int(row.n_src) + 1):
                spk, _, _, gender = source_fields(row, k)
                self.genders[spk] = gender
        # a reference clip needs a second utterance of the same speaker
        self.speakers = sorted(s for s, utts in self.pool.items() if len(utts) >= 2)
        log.info("Demo pool: %d speakers from %s/%s", len(self.speakers), self.root, split)

    @classmethod
    def from_demo_pack(cls, folder: str | Path) -> "SpeakerPool":
        """Pool from a demo pack (``speakers.csv`` + clips, see scripts/make_demo_pack.py)."""
        folder = Path(folder)
        self = cls.__new__(cls)
        self.root = folder
        self.pool, self.genders = {}, {}
        with open(folder / "speakers.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.pool.setdefault(row["speaker"], []).append(
                    (row["utt"], row["path"], int(row["length"])))
                self.genders[row["speaker"]] = row["gender"]
        self.speakers = sorted(s for s, utts in self.pool.items() if len(utts) >= 2)
        if len(self.speakers) < 4:
            raise ValueError(f"{folder}: need at least 4 speakers with 2+ clips")
        log.info("Demo pool: %d speakers from the demo pack %s", len(self.speakers), folder)
        return self

    def random_speakers(self, n: int, rng: np.random.Generator,
                        gender_mode: str = "any") -> list[dict[str, Any]]:
        """``n`` distinct speakers, each with one utterance and a different reference
        utterance. ``gender_mode``: ``any``, ``same`` (all one gender, the hard case) or
        ``mixed`` (at least one speaker of each gender)."""
        if gender_mode not in ("any", "same", "mixed"):
            raise ValueError(f"gender_mode must be any|same|mixed, got {gender_mode!r}")
        first = self.speakers[int(rng.integers(len(self.speakers)))]
        chosen = [first]
        g0 = self.genders.get(first, "")
        while len(chosen) < n:
            cands = [s for s in self.speakers if s not in chosen]
            if gender_mode == "same":
                cands = [s for s in cands if self.genders.get(s, "") == g0] or cands
            elif gender_mode == "mixed" and len(chosen) == n - 1 and \
                    len({self.genders.get(s, "") for s in chosen}) == 1:
                cands = [s for s in cands if self.genders.get(s, "") != g0] or cands
            chosen.append(cands[int(rng.integers(len(cands)))])
        out = []
        for spk in chosen:
            utts = self.pool[spk]
            i, j = rng.choice(len(utts), size=2, replace=False)
            utt, path, _ = utts[int(i)]
            ref_utt, ref_path, _ = utts[int(j)]
            out.append({
                "speaker": spk, "gender": self.genders.get(spk, "?"),
                "utt": utt, "audio": read_audio(self.root / path),
                "ref_utt": ref_utt, "ref_audio": read_audio(self.root / ref_path),
            })
        return out


def load_speaker_pool(data_cfg: dict[str, Any], split: str = "test",
                      demo_pack: str | Path | None = None) -> SpeakerPool | None:
    """The split's speakers if the dataset is present, else the bundled demo pack, else None."""
    if Path(data_cfg["root"]).is_dir():
        try:
            return SpeakerPool(data_cfg, split)
        except (FileNotFoundError, OSError, KeyError, ValueError) as e:
            log.info("Dataset not usable for random speakers (%s); trying the demo pack", e)
    else:
        log.info("No dataset at %s; using the demo pack for random speakers", data_cfg["root"])
    pack = Path(demo_pack) if demo_pack else Path(__file__).resolve().parents[2] / "demo_assets"
    try:
        return SpeakerPool.from_demo_pack(pack)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        log.warning("Random test speakers disabled: no dataset and no demo pack (%s)", e)
        return None
