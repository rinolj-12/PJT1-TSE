"""On-the-fly (dynamic) mixing of 2-4 speaker training mixtures for TSE.

Every access builds a fresh mixture: the target utterance of a training sample plus 1-3
interfering utterances of other training speakers, each with a random level and (optionally)
only partial overlap. Sources come from the clean training utterance pool (the Libri2Mix
``s1``/``s2`` files of the training split), so no new audio is needed and the train split
stays speaker-disjoint from dev/test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dataset import TSEDataset, fit_length_pad, read_audio
from .enrollment import Pool, enrollment_candidates


def active_rms(x: np.ndarray, thresh: float = 1e-6) -> float:
    """RMS over the non-silent (non-zero-padded) samples of ``x``."""
    a = x[np.abs(x) > thresh]
    return float(np.sqrt(np.mean(a.astype(np.float64) ** 2))) if len(a) else 0.0


class DynamicMixDataset(TSEDataset):
    """Training dataset that mixes a random number of speakers per item.

    Args:
        samples: Training sample table; each row provides one target utterance.
        root, pool, spk2label, sample_rate, segment_seconds, enroll_seconds: as in
            :class:`TSEDataset` (train mode).
        n_spk_probs: Probabilities of 2, 3, 4, ... speakers in a mixture (target included).
        level_db_range: Each source gets an i.i.d. level ~ U(lo, hi) dB; interferers are
            scaled to ``level_i - level_target`` dB relative to the target's active RMS.
        partial_overlap_prob: Probability that an interferer is active only on a random
            sub-window of the segment.
        min_overlap: Minimum fraction of the segment covered by a partial interferer.
        max_peak: Mixtures with a larger peak are rescaled (target and interferers alike).
        similar_speakers: Optional ``speaker -> [most similar training speakers]`` (by ECAPA
            similarity, see :mod:`tse.speaker_similarity`).
        hard_pair_prob: Probability that a mixture draws its interferers from the target's
            ``similar_speakers`` instead of uniformly (needs ``similar_speakers``).

    Items also contain ``interferers`` (K, T): each scaled interferer as it is in the
    mixture, zero rows after the first ``n_interferers`` (K = the largest mixture size - 1).
    """

    def __init__(self, samples: pd.DataFrame, root: str | Path, pool: Pool,
                 spk2label: dict[str, int], sample_rate: int = 8000,
                 segment_seconds: float = 3.0, enroll_seconds: float = 4.0,
                 n_spk_probs: list[float] | tuple[float, ...] = (0.4, 0.35, 0.25),
                 level_db_range: tuple[float, float] = (-5.0, 5.0),
                 partial_overlap_prob: float = 0.25, min_overlap: float = 0.5,
                 max_peak: float = 0.95,
                 similar_speakers: dict[str, list[str]] | None = None,
                 hard_pair_prob: float = 0.0) -> None:
        super().__init__(samples, root, pool, spk2label, train=True, sample_rate=sample_rate,
                         segment_seconds=segment_seconds, enroll_seconds=enroll_seconds)
        probs = torch.tensor(n_spk_probs, dtype=torch.float64)
        if len(probs) == 0 or (probs < 0).any() or probs.sum() <= 0:
            raise ValueError(f"Invalid n_spk_probs {n_spk_probs}")
        self.n_spk_probs = probs / probs.sum()
        self.speakers = sorted(pool.keys())
        if len(self.speakers) < len(n_spk_probs) + 1:
            raise ValueError("Not enough training speakers for the requested mixture sizes")
        self.level_lo, self.level_hi = level_db_range
        self.partial_p = partial_overlap_prob
        self.min_overlap = min_overlap
        self.max_peak = max_peak
        if hard_pair_prob > 0 and not similar_speakers:
            raise ValueError("hard_pair_prob > 0 needs similar_speakers")
        self.similar = similar_speakers or {}
        self.hard_p = hard_pair_prob
        self.max_interferers = len(self.n_spk_probs)

    def _uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand(1).item())

    def _crop(self, path: str, length: int) -> np.ndarray:
        seg = self.segment
        start = self._randint(length - seg + 1) if length > seg else 0
        return fit_length_pad(read_audio(self.root / path, start, seg), seg)

    def _target_crop(self, path: str, length: int, tries: int = 3) -> np.ndarray:
        """Random crop of the target, re-drawn (up to ``tries`` times) if nearly silent."""
        x = self._crop(path, length)
        for _ in range(tries - 1):
            if active_rms(x) > 1e-3:
                break
            x = self._crop(path, length)
        return x

    def _partial(self, x: np.ndarray) -> np.ndarray:
        """Keep a random sub-window covering ``[min_overlap, 1]`` of the segment."""
        n = len(x)
        act = int(round(self._uniform(self.min_overlap, 1.0) * n))
        start = self._randint(n - act + 1)
        out = np.zeros_like(x)
        out[start:start + act] = x[start:start + act]
        return out

    def num_speakers(self) -> int:
        """Draw the number of speakers (target included) for one mixture."""
        return 2 + int(torch.multinomial(self.n_spk_probs, 1).item())

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self._records[i]
        n_spk = self.num_speakers()
        tgt = self._target_crop(r["target_path"], int(r["length"]))
        t_rms = max(active_rms(tgt), 1e-4)
        t_level = self._uniform(self.level_lo, self.level_hi)

        hard = self.hard_p > 0 and float(torch.rand(1).item()) < self.hard_p
        if hard:
            others = self.similar[r["speaker_id"]]
        else:
            others = [s for s in self.speakers if s != r["speaker_id"]]
        perm = torch.randperm(len(others))[:n_spk - 1].tolist()
        interf = np.zeros_like(tgt)
        intf_utts, intf_srcs = [], []
        for j in perm:
            spk = others[j]
            utt, path, length = self.pool[spk][self._randint(len(self.pool[spk]))]
            x = self._crop(path, length)
            if float(torch.rand(1).item()) < self.partial_p:
                x = self._partial(x)
            x_rms = active_rms(x)
            if x_rms < 1e-4:  # silent crop: contributes nothing, keep it silent
                continue
            rel_db = self._uniform(self.level_lo, self.level_hi) - t_level
            x = x * (t_rms / x_rms) * 10 ** (rel_db / 20)
            interf += x
            intf_utts.append(utt)
            intf_srcs.append(x)

        mix = tgt + interf
        peak = float(np.abs(mix).max())
        if peak > self.max_peak:
            scale = self.max_peak / peak
            mix, tgt = mix * scale, tgt * scale
            intf_srcs = [x * scale for x in intf_srcs]
        srcs = np.zeros((self.max_interferers, len(mix)), dtype=np.float32)
        for k, x in enumerate(intf_srcs):
            srcs[k] = x
        cands = enrollment_candidates(self.pool, r["speaker_id"], r["utt_id"])
        e_utt, e_path, e_len = cands[self._randint(len(cands))]
        enr = self._load_enrollment(e_path, e_len)
        mix_t = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
        tgt_t = torch.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32))
        meta = {
            "mixture_id": "dm:" + "_".join([r["utt_id"]] + intf_utts), "role": "s1",
            "speaker_id": r["speaker_id"], "utt_id": r["utt_id"], "enroll_utt": e_utt,
            "enroll_path": e_path, "target_path": r["target_path"], "length": len(mix),
            "same_gender": "", "n_src": n_spk, "hard_pair": hard,
        }
        return {
            "mixture": mix_t,
            "enrollment": torch.from_numpy(np.ascontiguousarray(enr)),
            "target": tgt_t,
            "interference": mix_t - tgt_t,
            "interferers": torch.from_numpy(srcs),
            "n_interferers": len(intf_srcs),
            "speaker_label": self.spk2label.get(r["speaker_id"], -1),
            "meta": meta,
        }
