"""Build (and cache) split indexes, utterance pools and fixed eval enrollments; print stats.

Example:
    python scripts/prepare_data.py --config configs/proposed.yaml --verify
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tse.config import load_config, resolve_paths  # noqa: E402
from tse.data.dataset import prepare_splits  # noqa: E402
from tse.data.index import verify_files  # noqa: E402
from tse.utils import setup_logging  # noqa: E402

log = logging.getLogger("prepare_data")


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/proposed.yaml")
    ap.add_argument("--data-root", default=None, help="Libri2Mix root (contains wav8k/)")
    ap.add_argument("--rebuild", action="store_true", help="ignore cached indexes")
    ap.add_argument("--verify", action="store_true", help="check every referenced file exists")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    setup_logging()
    cfg = resolve_paths(load_config(args.config, args.set), args.data_root)
    d = cfg["data"]
    log.info("Dataset root: %s", d["root"])
    data = prepare_splits(d, rebuild=args.rebuild)

    print(f"\n{'split':<6} {'name':<10} {'mixtures':>8} {'samples':>8} {'dropped':>7} "
          f"{'speakers':>8} {'utts':>6} {'len s min/med/max':>20} {'hours':>6}")
    for k, name in (("train", d["train_split"]), ("val", d["val_split"]), ("test", d["test_split"])):
        idx = data.indexes[k]
        L = idx["length"].to_numpy() / d["sample_rate"]
        n_utt = sum(len(v) for v in data.pools[k].values())
        print(f"{k:<6} {name:<10} {len(idx):>8} {len(data.samples[k]):>8} {data.dropped[k]:>7} "
              f"{len(data.pools[k]):>8} {n_utt:>6} "
              f"{L.min():>6.2f}/{np.median(L):>5.2f}/{L.max():>5.2f} {L.sum() / 3600:>6.1f}")
        utt_per_spk = [len(v) for v in data.pools[k].values()]
        print(f"       utterances per speaker: min {min(utt_per_spk)}  median "
              f"{int(np.median(utt_per_spk))}  max {max(utt_per_spk)}")
        sg = data.samples[k]["same_gender"]
        if (sg != "").any():
            print(f"       same-gender samples: {(sg == 'True').mean() * 100:.1f}%")

    spk = {k: set(v.keys()) for k, v in data.pools.items()}
    overlaps = {f"{a}/{b}": len(spk[a] & spk[b])
                for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    print(f"\nSpeaker overlap between splits: {overlaps}")
    if any(overlaps.values()):
        log.warning("Splits are NOT speaker-disjoint!")
    else:
        print("Train/val/test speakers are disjoint.")

    for k in ("val", "test"):
        t = data.samples[k]
        bad = int((t["enroll_utt"] == t["utt_id"]).sum())
        wrong_spk = int((t["enroll_utt"].str.split("-").str[0] != t["speaker_id"]).sum())
        print(f"{k}: fixed enrollments={len(t)}, enrollment==target utt: {bad}, "
              f"enrollment from wrong speaker: {wrong_spk}")
    print(f"Training speakers (classifier classes): {len(data.spk2label)}")

    r = data.indexes["train"].iloc[0]
    root = Path(d["root"])
    info = sf.info(str(root / r["mix_path"]))
    print(f"\nAudio check ({r['mixture_id']}): {info.samplerate} Hz, {info.subtype}, "
          f"{info.channels} ch, {info.frames} frames (index says {r['length']})")
    if info.samplerate != d["sample_rate"]:
        log.error("Sample rate mismatch: file %d vs config %d", info.samplerate, d["sample_rate"])

    if args.verify:
        for k, idx in data.indexes.items():
            missing = verify_files(root, idx)
            print(f"verify {k}: {missing} missing files out of {3 * len(idx)}")
    print(f"\nCaches written to {d['cache_dir']}")


if __name__ == "__main__":
    main()
