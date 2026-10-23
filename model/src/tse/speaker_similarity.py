"""Speaker-level ECAPA embeddings and voice-similarity lookups.

Used to draw "hard" interferers that sound like the target during training
(``data.dynamic_mix.hard_pair_prob``) and to break results down by how similar the target
and the interferers sound (``scripts/similarity_breakdown.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import read_audio
from .data.enrollment import Pool

log = logging.getLogger(__name__)


@torch.no_grad()
def speaker_embeddings(pool: Pool, root: str | Path, device: torch.device, n_utts: int = 5,
                       seconds: float = 6.0, sample_rate: int = 8000,
                       seed: int = 0) -> dict[str, np.ndarray]:
    """Unit-norm mean ECAPA embedding of up to ``n_utts`` utterances (center crops of
    ``seconds``) per speaker, utterances chosen with a fixed seed."""
    from .models.speaker_encoder import FrozenECAPA

    ecapa = FrozenECAPA(sample_rate).to(device)
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    out = {}
    for spk in sorted(pool):
        utts = pool[spk]
        pick = rng.choice(len(utts), size=min(n_utts, len(utts)), replace=False)
        embs = []
        for i in sorted(pick):
            x = read_audio(Path(root) / utts[int(i)][1])
            if len(x) > n:
                s = (len(x) - n) // 2
                x = x[s:s + n]
            e = ecapa.ecapa_embedding(torch.from_numpy(x)[None].to(device))
            embs.append(F.normalize(e, dim=-1)[0])
        m = F.normalize(torch.stack(embs).mean(0), dim=0)
        out[spk] = m.cpu().numpy()
    return out


def load_or_compute_speaker_embeddings(cache_path: str | Path, pool: Pool, root: str | Path,
                                       device: torch.device, **kw) -> dict[str, np.ndarray]:
    """:func:`speaker_embeddings`, cached in ``cache_path`` (recomputed if speakers differ)."""
    cache_path = Path(cache_path)
    if cache_path.is_file():
        emb = torch.load(cache_path, weights_only=False)
        if set(emb) == set(pool):
            return emb
        log.info("Speaker set changed; recomputing %s", cache_path)
    log.info("Computing ECAPA speaker embeddings for %d speakers -> %s", len(pool), cache_path)
    emb = speaker_embeddings(pool, root, device, **kw)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(emb, cache_path)
    return emb


def most_similar(emb: dict[str, np.ndarray], k: int) -> dict[str, list[str]]:
    """For every speaker, the ``k`` other speakers with the highest cosine similarity."""
    spks = sorted(emb)
    M = np.stack([emb[s] for s in spks])
    S = M @ M.T
    np.fill_diagonal(S, -np.inf)
    order = np.argsort(-S, axis=1)[:, :k]
    return {s: [spks[j] for j in order[i]] for i, s in enumerate(spks)}
