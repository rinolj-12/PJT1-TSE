"""Break test results down by how similar the target and the interferers sound.

For every evaluated test sample, the voice similarity is the ECAPA cosine between the
target speaker and the most similar interferer (speaker-level mean embeddings). Samples are
split into quintiles of that similarity (the same bins for every run), and mean SI-SDRi and
the wrong-speaker rate (NSR, SI-SDRi < 0) are reported per bin and run.

Example:
    python scripts/similarity_breakdown.py saved_models/multispk_r6_ep79 \\
        experiments/ecapa_judge/control experiments/ecapa_judge/judge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.data.enrollment import build_utterance_pool  # noqa: E402
from tse.data.index import load_index  # noqa: E402
from tse.speaker_similarity import load_or_compute_speaker_embeddings  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("similarity_breakdown")

SETS = [("test", "2 spk"), ("Libri3Mix_test", "3 spk"), ("Libri4Mix_test", "4 spk")]
BINS = ["1 least similar", "2", "3", "4", "5 most similar"]


def max_similarity(df: pd.DataFrame, emb: dict[str, np.ndarray]) -> np.ndarray:
    """ECAPA cosine of the target speaker vs. its most similar interferer, per row."""
    out = np.empty(len(df))
    for i, (mid, spk) in enumerate(zip(df["mixture_id"], df["speaker_id"].astype(str))):
        others = [u.split("-")[0] for u in mid.split("_")]
        others.remove(spk)
        out[i] = max(float(emb[spk] @ emb[o]) for o in others)
    return out


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories containing *_per_utt.csv")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "proposed.yaml"))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="optional JSON file for the table")
    args = ap.parse_args()
    setup_logging()

    d = resolve_paths(load_config(args.config), args.data_root)["data"]
    pool = build_utterance_pool(load_index(d["root"], d["test_split"], d))
    emb = load_or_compute_speaker_embeddings(Path(d["cache_dir"]) / f"ecapa_spk_{d['test_split']}.pt",
                                             pool, d["root"], resolve_device(args.device),
                                             sample_rate=d["sample_rate"])
    result = {}
    for prefix, label in SETS:
        frames = {}
        for run in args.runs:
            f = Path(run) / f"{prefix}_per_utt.csv"
            if f.is_file():
                frames[Path(run).name] = pd.read_csv(f)
        if not frames:
            continue
        ref = next(iter(frames.values()))
        sim = max_similarity(ref, emb)
        edges = np.quantile(sim, [0.2, 0.4, 0.6, 0.8])
        print(f"\n{label} ({len(ref)} samples): ECAPA similarity of target vs closest interferer")
        print(f"{'bin':16s} {'sim':>5s}" + "".join(f"  {n[:22]:>22s}" for n in frames))
        result[label] = {}
        for b, name in enumerate(BINS):
            row = f"{name:16s}"
            for k, (run, df) in enumerate(frames.items()):
                if len(df) != len(ref) or not (df["mixture_id"].values == ref["mixture_id"].values).all():
                    raise SystemExit(f"{run}/{prefix}: samples differ from {next(iter(frames))}")
                s = max_similarity(df, emb) if k else sim
                m = np.digitize(s, edges) == b
                imp = df["si_sdri"].values[m]
                if k == 0:
                    row += f" {s[m].mean():5.2f}"
                row += f"  {imp.mean():7.2f} dB  NSR {100 * (imp < 0).mean():4.1f}%"
                result[label].setdefault(run, {})[name] = {
                    "si_sdri": float(imp.mean()), "nsr": float((imp < 0).mean()), "n": int(m.sum())}
            print(row)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
