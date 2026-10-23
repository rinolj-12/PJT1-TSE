"""Build and cache per-split mixture indexes from LibriMix metadata.

The metadata CSVs shipped with LibriMix contain absolute paths from the machine
that generated the data, so they are *not* trusted. Paths are rebuilt from the
dataset root and the mixture ID:

    {root}/wav{sr//1000}k/{mode}/{split}/{mix_dir|s1|s2|...}/{mixture_ID}.wav

Mixtures may contain any number of speakers (Libri2Mix, Libri3Mix, our Libri4Mix-like
sets); the number of sources is read from the mixture ID. Per-source index columns are
``s{k}`` (path), ``spk{k}``, ``utt{k}``, ``gender{k}`` for k = 1..n_src.

Index paths are stored relative to ``root`` so cached indexes stay valid if the
dataset is moved.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BASE_COLUMNS = ["mixture_id", "mix_path", "length", "n_src"]


def source_columns(n_src: int) -> list[str]:
    """Per-source index columns for ``n_src`` sources."""
    return [f"{c}{k}" for k in range(1, n_src + 1) for c in ("s", "spk", "utt", "gender")]


def index_n_src(index: pd.DataFrame) -> int:
    """Maximum number of sources per mixture in an index."""
    return int(index["n_src"].max())


def source_fields(row: Any, k: int) -> tuple[str, str, str, str]:
    """``(spk, utt, path, gender)`` of source ``k`` (1-based) of an index row (itertuples)."""
    return (str(getattr(row, f"spk{k}")), getattr(row, f"utt{k}"), getattr(row, f"s{k}"),
            getattr(row, f"gender{k}"))


def read_index_csv(path: str | Path) -> pd.DataFrame:
    """Read an index CSV with speaker/gender columns kept as strings."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ("length", "n_src"):
        if c in df.columns:  # absent in pre-N-speaker caches, which load_index rebuilds
            df[c] = df[c].astype(int)
    return df


def split_dir(root: Path, sample_rate: int, mode: str) -> Path:
    """Return ``{root}/wav8k/min``-style directory for the given settings."""
    return Path(root) / f"wav{sample_rate // 1000}k" / mode


def mix_dirname(mix_type: str) -> str:
    """Map config ``mix_type`` to the LibriMix folder name."""
    return {"clean": "mix_clean", "both": "mix_both"}[mix_type]


def parse_mixture_id(mixture_id: str) -> list[tuple[str, str]]:
    """Parse ``'{spk-chap-utt}_{spk-chap-utt}[_...]'`` into ``[(spk, utt), ...]``."""
    parts = mixture_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected LibriMix mixture ID: {mixture_id}")
    out = []
    for utt in parts:
        fields = utt.split("-")
        if len(fields) != 3:
            raise ValueError(f"Unexpected LibriSpeech utterance ID '{utt}' in {mixture_id}")
        out.append((fields[0], utt))
    return out


def load_speaker_genders(gender_file: str | Path | None, root: Path) -> dict[str, str] | None:
    """Read speaker gender from LibriSpeech ``SPEAKERS.TXT``.

    If ``gender_file`` is None, look in ``root``, ``root/../LibriSpeech`` and
    ``root/../../data/LibriSpeech`` (for datasets generated inside the project).
    Returns ``None`` (with a warning) if no file is found.
    """
    root = Path(root)
    candidates = [Path(gender_file)] if gender_file else [
        root / "SPEAKERS.TXT",
        root.parent / "LibriSpeech" / "SPEAKERS.TXT",
        root.parent.parent.parent / "data" / "LibriSpeech" / "SPEAKERS.TXT",
    ]
    for c in candidates:
        if c.is_file():
            genders: dict[str, str] = {}
            with open(c, encoding="utf-8") as f:
                for line in f:
                    if line.startswith(";") or "|" not in line:
                        continue
                    fields = [x.strip() for x in line.split("|")]
                    genders[fields[0]] = fields[1]
            log.info("Loaded gender for %d speakers from %s", len(genders), c)
            return genders
    log.warning("No SPEAKERS.TXT found (tried %s); gender breakdown will be skipped.",
                [str(c) for c in candidates])
    return None


def build_index(root: str | Path, split: str, sample_rate: int = 8000, mode: str = "min",
                mix_type: str = "clean", genders: dict[str, str] | None = None) -> pd.DataFrame:
    """Build the mixture index of one split from ``metadata/mixture_{split}_{mix_dir}.csv``."""
    root = Path(root)
    sdir = split_dir(root, sample_rate, mode)
    mdir = mix_dirname(mix_type)
    meta = sdir / "metadata" / f"mixture_{split}_{mdir}.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"Metadata not found: {meta}")
    if not (sdir / split / mdir).is_dir():
        raise FileNotFoundError(
            f"Mixture folder {sdir / split / mdir} is missing (mix_type={mix_type!r}).")
    df = pd.read_csv(meta)
    for col in ("mixture_ID", "length"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' missing in {meta}; found {list(df.columns)}")

    rel = Path(sdir.relative_to(root)) / split
    rows: list[dict[str, Any]] = []
    n_dup = 0
    for mid, length in zip(df["mixture_ID"], df["length"]):
        srcs = parse_mixture_id(mid)
        row: dict[str, Any] = {"mixture_id": mid, "mix_path": (rel / mdir / f"{mid}.wav").as_posix(),
                               "length": int(length), "n_src": len(srcs)}
        for k, (spk, utt) in enumerate(srcs, start=1):
            row[f"s{k}"] = (rel / f"s{k}" / f"{mid}.wav").as_posix()
            row[f"spk{k}"] = spk
            row[f"utt{k}"] = utt
            row[f"gender{k}"] = genders.get(spk, "") if genders else ""
        n_dup += len({s for s, _ in srcs}) < len(srcs)
        rows.append(row)
    n_max = max(r["n_src"] for r in rows)
    out = pd.DataFrame(rows, columns=BASE_COLUMNS + source_columns(n_max)).fillna("")
    if n_dup:
        log.warning("%s: %d mixtures contain the same speaker twice", split, n_dup)
    return out


def verify_files(root: str | Path, index: pd.DataFrame) -> int:
    """Return the number of missing audio files referenced by ``index``."""
    root = Path(root)
    missing = 0
    cols = ["mix_path"] + [f"s{k}" for k in range(1, index_n_src(index) + 1)]
    for col in cols:
        for p in index[col]:
            if p and not (root / p).is_file():
                missing += 1
                if missing <= 5:
                    log.error("Missing file: %s", root / p)
    return missing


def make_speaker_disjoint_splits(index: pd.DataFrame, ratios: list[float],
                                 seed: int) -> dict[str, pd.DataFrame]:
    """Split one index into speaker-disjoint train/val/test parts (by speaker, not mixture).

    Mixtures whose speakers end up in different parts are dropped.
    """
    n = index_n_src(index)
    speakers = sorted(set().union(*[set(index[f"spk{k}"]) for k in range(1, n + 1)]) - {""})
    rng = np.random.default_rng(seed)
    perm = [speakers[i] for i in rng.permutation(len(speakers))]
    n_tr = int(round(ratios[0] * len(perm)))
    n_va = int(round(ratios[1] * len(perm)))
    assign = {s: "train" for s in perm[:n_tr]}
    assign.update({s: "val" for s in perm[n_tr:n_tr + n_va]})
    assign.update({s: "test" for s in perm[n_tr + n_va:]})
    a1 = index["spk1"].map(assign)
    keep = pd.Series(True, index=index.index)
    for k in range(2, n + 1):
        keep &= (index[f"spk{k}"].map(assign) == a1) | (index[f"spk{k}"] == "")
    log.info("Custom split: dropped %d/%d cross-partition mixtures", int((~keep).sum()), len(index))
    return {k: index[keep & (a1 == k)].reset_index(drop=True) for k in ("train", "val", "test")}


def dataset_tag(root: str | Path) -> str:
    """Cache-file prefix: '' for Libri2Mix (keeps the original names), else ``'{folder}_'``."""
    name = Path(root).name
    return "" if name == "Libri2Mix" else f"{name}_"


def _cache_path(cache_dir: Path, split: str, data_cfg: dict[str, Any], root: str | Path) -> Path:
    return cache_dir / (f"index_{dataset_tag(root)}{split}_{data_cfg['sample_rate']}_"
                        f"{data_cfg['mode']}_{data_cfg['mix_type']}.csv")


def load_index(root: str | Path, split: str, data_cfg: dict[str, Any], rebuild: bool = False,
               genders: dict[str, str] | None = None) -> pd.DataFrame:
    """Build (or load from cache) the index of one split of the dataset at ``root``."""
    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(cache_dir, split, data_cfg, root)
    if cp.is_file() and not rebuild:
        idx = read_index_csv(cp)
        if "n_src" in idx.columns and "s1" in idx.columns:
            return idx
        log.info("Cached index %s has an old format; rebuilding", cp)
    if genders is None:
        genders = load_speaker_genders(data_cfg.get("gender_file"), Path(root)) or {}
    idx = build_index(root, split, data_cfg["sample_rate"], data_cfg["mode"], data_cfg["mix_type"],
                      genders)
    idx.to_csv(cp, index=False)
    log.info("Cached index %s (%d mixtures)", cp, len(idx))
    return idx


def load_split_indexes(data_cfg: dict[str, Any], rebuild: bool = False) -> dict[str, pd.DataFrame]:
    """Return ``{'train', 'val', 'test'}`` indexes, building and caching them as needed.

    If the validation/test metadata are missing, a speaker-disjoint 80/10/10 split of
    the training split is created and saved to ``data.splits_dir``.
    """
    root = Path(data_cfg["root"])
    genders: dict[str, str] | None = None

    def get(split: str) -> pd.DataFrame:
        nonlocal genders
        cp = _cache_path(Path(data_cfg["cache_dir"]), split, data_cfg, root)
        if genders is None and (rebuild or not cp.is_file()):
            genders = load_speaker_genders(data_cfg.get("gender_file"), root) or {}
        return load_index(root, split, data_cfg, rebuild, genders)

    sdir = split_dir(root, data_cfg["sample_rate"], data_cfg["mode"]) / "metadata"
    mdir = mix_dirname(data_cfg["mix_type"])
    have_eval = all((sdir / f"mixture_{data_cfg[k]}_{mdir}.csv").is_file()
                    for k in ("val_split", "test_split"))
    if have_eval:
        return {"train": get(data_cfg["train_split"]), "val": get(data_cfg["val_split"]),
                "test": get(data_cfg["test_split"])}

    log.warning("Validation/test metadata missing: creating a speaker-disjoint custom split.")
    splits_dir = Path(data_cfg["splits_dir"])
    files = {k: splits_dir / f"{k}.csv" for k in ("train", "val", "test")}
    if all(f.is_file() for f in files.values()) and not rebuild:
        return {k: read_index_csv(f) for k, f in files.items()}
    parts = make_speaker_disjoint_splits(get(data_cfg["train_split"]),
                                         data_cfg["custom_split_ratios"], seed=0)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for k, df in parts.items():
        df.to_csv(files[k], index=False)
    return parts
