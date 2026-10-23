"""Generate 3- and 4-speaker evaluation sets (clean, 8 kHz, min mode) from LibriSpeech.

* **Libri3Mix** (official): rebuilt exactly like the LibriMix generator from its metadata
  (``source * gain`` at 16 kHz -> ``resample_poly`` to 8 kHz -> truncate to the shortest
  source -> sum). Noise is not added (``mix_clean``).
* **Libri4Mix** (ours, not an official benchmark): each Libri3Mix mixture plus a 4th
  speaker from the same LibriSpeech subset (a speaker not already in the mixture), drawn
  with a fixed seed per mixture. Its level is uniform between the quietest and loudest of the
  other three (RMS at 16 kHz). The recipe is saved to ``metadata/libri4mix_{split}_recipe.csv``.

Output layout mirrors Libri2Mix, so the normal index / evaluation code works unchanged:
``{out}/Libri{N}Mix/wav8k/min/{split}/{mix_clean,s1..sN}/{mixture_ID}.wav`` and
``{out}/Libri{N}Mix/wav8k/min/metadata/mixture_{split}_mix_clean.csv``.

The input dataset is only read.

Example:
    python scripts/make_multispk_sets.py --sets 3 4 --splits dev test
"""

from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from tse.config import PROJECT_ROOT  # noqa: E402
from tse.utils import setup_logging  # noqa: E402

log = logging.getLogger("make_multispk_sets")
SUBSETS = {"dev": "dev-clean", "test": "test-clean", "train-100": "train-clean-100"}
IN_RATE, OUT_RATE = 16000, 8000


def rms_db(x: np.ndarray) -> float:
    """RMS level in dB."""
    return float(10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12))


def utt_id(path: str) -> str:
    """LibriSpeech utterance ID from a ``subset/spk/chap/spk-chap-utt.flac`` path."""
    return Path(path).stem


def list_speaker_utterances(librispeech: Path, subset: str) -> dict[str, list[str]]:
    """``speaker -> sorted relative flac paths`` for one LibriSpeech subset."""
    out: dict[str, list[str]] = {}
    for f in sorted((librispeech / subset).glob("*/*/*.flac")):
        out.setdefault(f.parent.parent.name, []).append(f.relative_to(librispeech).as_posix())
    return out


def add_fourth_source(recipe: pd.DataFrame, librispeech: Path, subset: str,
                      seed: int) -> pd.DataFrame:
    """Extend a Libri3Mix recipe with a seeded 4th speaker per mixture."""
    spk_utts = list_speaker_utterances(librispeech, subset)
    speakers = sorted(spk_utts)
    rows = []
    for i, r in enumerate(recipe.itertuples(index=False)):
        rng = np.random.default_rng([seed, i])
        present = {utt_id(getattr(r, f"source_{k}_path")).split("-")[0] for k in (1, 2, 3)}
        cands = [s for s in speakers if s not in present]
        spk = cands[int(rng.integers(len(cands)))]
        path = spk_utts[spk][int(rng.integers(len(spk_utts[spk])))]
        levels = []
        for k in (1, 2, 3):
            x, _ = sf.read(str(librispeech / getattr(r, f"source_{k}_path")), dtype="float32")
            levels.append(rms_db(x * getattr(r, f"source_{k}_gain")))
        x4, _ = sf.read(str(librispeech / path), dtype="float32")
        level = rng.uniform(min(levels), max(levels))
        gain = float(10 ** ((level - rms_db(x4)) / 20))
        d = r._asdict()
        d.update(source_4_path=path, source_4_gain=gain)
        d["mixture_ID"] = f"{r.mixture_ID}_{utt_id(path)}"
        rows.append(d)
    return pd.DataFrame(rows)


def make_mixture(args: tuple) -> dict:
    """Create and write one mixture (runs in a worker process)."""
    row, n_src, librispeech, split_dir = args
    srcs = []
    for k in range(1, n_src + 1):
        x, sr = sf.read(str(Path(librispeech) / row[f"source_{k}_path"]), dtype="float32")
        if sr != IN_RATE:
            raise ValueError(f"Expected {IN_RATE} Hz LibriSpeech audio, got {sr}")
        srcs.append(resample_poly(x * row[f"source_{k}_gain"], OUT_RATE, IN_RATE))
    n = min(len(s) for s in srcs)
    srcs = [s[:n] for s in srcs]
    mix = np.sum(srcs, axis=0)
    mid = row["mixture_ID"]
    out = {"mixture_ID": mid, "mixture_path": f"mix_clean/{mid}.wav", "length": n}
    sf.write(str(Path(split_dir) / "mix_clean" / f"{mid}.wav"), mix, OUT_RATE)
    for k, s in enumerate(srcs, start=1):
        sf.write(str(Path(split_dir) / f"s{k}" / f"{mid}.wav"), s, OUT_RATE)
        out[f"source_{k}_path"] = f"s{k}/{mid}.wav"
    return out


def build_set(n_src: int, split: str, args: argparse.Namespace) -> None:
    """Generate one (n_src, split) set."""
    subset = SUBSETS[split]
    root = Path(args.out) / f"Libri{n_src}Mix" / "wav8k" / "min"
    meta_dir = root / "metadata"
    meta_csv = meta_dir / f"mixture_{split}_mix_clean.csv"
    if meta_csv.is_file() and not args.overwrite:
        log.info("%s exists, skipping (use --overwrite to rebuild)", meta_csv)
        return
    recipe = pd.read_csv(Path(args.metadata) / f"libri3mix_{subset}.csv")
    if args.max_mixtures:
        recipe = recipe.iloc[:args.max_mixtures]
    if n_src == 4:
        log.info("Drawing 4th speakers for %d %s mixtures", len(recipe), split)
        recipe = add_fourth_source(recipe, Path(args.librispeech), subset, args.seed)
    elif n_src != 3:
        raise ValueError("Only 3- and 4-speaker sets are supported")
    meta_dir.mkdir(parents=True, exist_ok=True)
    recipe.to_csv(meta_dir / f"libri{n_src}mix_{split}_recipe.csv", index=False)
    split_dir = root / split
    for d in ["mix_clean"] + [f"s{k}" for k in range(1, n_src + 1)]:
        (split_dir / d).mkdir(parents=True, exist_ok=True)

    cols = ["mixture_ID"] + [f"source_{k}_{c}" for k in range(1, n_src + 1)
                             for c in ("path", "gain")]
    jobs = [({c: r[c] for c in cols}, n_src, str(args.librispeech), str(split_dir))
            for r in recipe.to_dict("records")]
    log.info("Writing %d Libri%dMix %s mixtures to %s", len(jobs), n_src, split, split_dir)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(make_mixture, jobs, chunksize=32))
    md = pd.DataFrame(rows)
    md = md[["mixture_ID", "mixture_path"] + [f"source_{k}_path" for k in range(1, n_src + 1)]
            + ["length"]]
    md.to_csv(meta_csv, index=False)
    L = md["length"] / OUT_RATE
    log.info("Libri%dMix %s: %d mixtures, length min/median/max %.2f/%.2f/%.2f s", n_src, split,
             len(md), L.min(), L.median(), L.max())


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--librispeech", default=str((PROJECT_ROOT / "../data/LibriSpeech").resolve()))
    ap.add_argument("--metadata", default=str(
        (PROJECT_ROOT / "../data/LibriMix_generator/metadata/Libri3Mix").resolve()))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "generated"))
    ap.add_argument("--sets", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--max-mixtures", type=int, default=None)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    setup_logging()
    for n in args.sets:
        for split in args.splits:
            build_set(n, split, args)


if __name__ == "__main__":
    main()
