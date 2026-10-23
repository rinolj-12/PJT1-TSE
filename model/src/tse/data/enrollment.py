"""Speaker utterance pools, TSE sample lists and enrollment selection.

Libri2Mix has no enrollment utterances. For each split we collect every clean
source file (``s1``/``s2``) of that split, deduplicated by LibriSpeech utterance
ID, into ``speaker -> [(utt_id, path, length)]``. The enrollment for a sample
with target speaker S and target utterance U is any utterance of S in the same
split whose ID differs from U.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .index import source_fields

log = logging.getLogger(__name__)

Pool = dict[str, list[tuple[str, str, int]]]


def build_utterance_pool(index: pd.DataFrame) -> Pool:
    """Build ``speaker -> sorted [(utt_id, rel_path, length)]``, deduplicated by utterance ID.

    When an utterance appears in several mixtures (dev/test), the copy from the
    lexicographically first mixture ID is kept, so the pool is deterministic.
    In ``min`` mode every source file has the mixture's length.
    """
    seen: dict[str, tuple[str, str, int]] = {}
    for row in index.sort_values("mixture_id").itertuples(index=False):
        for k in range(1, int(row.n_src) + 1):
            spk, utt, path, _ = source_fields(row, k)
            if utt not in seen:
                seen[utt] = (spk, path, int(row.length))
    pool: Pool = {}
    for utt, (spk, path, length) in seen.items():
        pool.setdefault(spk, []).append((utt, path, length))
    for spk in pool:
        pool[spk].sort()
    return pool


def build_samples(index: pd.DataFrame, pool: Pool) -> tuple[pd.DataFrame, int]:
    """Expand each mixture into one TSE sample per source (target = s1, s2, ...).

    ``same_gender`` is True if at least one interfering speaker has the target's gender
    (for 2-speaker mixtures this is the usual same/different-gender split).
    Samples whose target speaker has no other utterance in the pool are dropped.
    Returns ``(samples, n_dropped)``.
    """
    rows = []
    dropped = 0
    for r in index.itertuples(index=False):
        n = int(r.n_src)
        srcs = [source_fields(r, k) for k in range(1, n + 1)]
        for k, (spk, utt, path, g) in enumerate(srcs, start=1):
            if not any(u != utt for u, _, _ in pool.get(spk, [])):
                dropped += 1
                continue
            others = [s for j, s in enumerate(srcs, start=1) if j != k]
            o_genders = [o[3] for o in others]
            same_gender = "" if not g or not all(o_genders) else str(g in o_genders)
            rows.append({
                "mixture_id": r.mixture_id, "role": f"s{k}", "mix_path": r.mix_path,
                "target_path": path, "length": int(r.length), "speaker_id": spk,
                "utt_id": utt, "other_speaker_id": "+".join(o[0] for o in others),
                "interferer_paths": "|".join(o[2] for o in others),
                "same_gender": same_gender, "n_src": n,
            })
    if dropped:
        log.warning("Dropped %d samples whose speaker has no other utterance", dropped)
    return pd.DataFrame(rows), dropped


def enrollment_candidates(pool: Pool, speaker: str, exclude_utt: str) -> list[tuple[str, str, int]]:
    """Utterances of ``speaker`` that are not ``exclude_utt``."""
    return [c for c in pool[speaker] if c[0] != exclude_utt]


def fixed_enrollment_table(samples: pd.DataFrame, pool: Pool, seed: int) -> pd.DataFrame:
    """Pick one enrollment per sample with a seeded RNG (for validation/test).

    Iterates samples in a canonical order (mixture_id, role) so the result does not
    depend on the row order of ``samples``.
    """
    rng = np.random.default_rng(seed)
    order = samples.sort_values(["mixture_id", "role"]).reset_index(drop=True)
    utts, paths, lengths = [], [], []
    for r in order.itertuples(index=False):
        cands = enrollment_candidates(pool, r.speaker_id, r.utt_id)
        u, p, n = cands[int(rng.integers(len(cands)))]
        utts.append(u)
        paths.append(p)
        lengths.append(n)
    order["enroll_utt"] = utts
    order["enroll_path"] = paths
    order["enroll_length"] = lengths
    return order


def load_or_create_enrollment_table(samples: pd.DataFrame, pool: Pool, seed: int,
                                    path: str | Path) -> pd.DataFrame:
    """Load ``enrollment_{split}.csv`` if it matches ``samples``, else create and save it.

    Saving the table guarantees every model is evaluated on identical
    (mixture, enrollment) pairs.
    """
    path = Path(path)
    if path.is_file():
        table = pd.read_csv(path, dtype={"speaker_id": str, "other_speaker_id": str,
                                         "same_gender": str}, keep_default_na=False)
        key = ["mixture_id", "role"]
        if len(table) == len(samples) and set(map(tuple, table[key].values)) == set(
                map(tuple, samples[key].values)):
            missing = [c for c in samples.columns if c not in table.columns]
            if missing:  # table written by an older version: re-attach new sample columns
                table = table.merge(samples[key + missing], on=key, how="left")
            return table
        log.warning("Enrollment table %s does not match the current samples; rebuilding", path)
    table = fixed_enrollment_table(samples, pool, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    log.info("Saved fixed enrollment table %s (%d rows)", path, len(table))
    return table
