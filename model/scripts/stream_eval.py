"""Quality vs latency of sliding-window streaming, against whole-utterance processing.

Runs a registered model over a fixed, seeded subset of the 2-/3-/4-speaker test sets (with
their fixed enrollments), once offline and once per streaming setting, and reports mean
SI-SDRi, the wrong-speaker rate (NSR) and model compute per hop. Use it to pick the
real-time defaults.

Example:
    python scripts/stream_eval.py --model r6_multispk --n 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.data.dataset import TSEDataset, prepare_eval_set, val_subset  # noqa: E402
from tse.metrics import si_sdr_np  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("stream_eval")

SETS = {"2spk": None, "3spk": "generated/Libri3Mix", "4spk": "generated/Libri4Mix"}
# (window s, hop s, lookahead s)
SETTINGS = [(3.0, 0.1, 0.0), (3.0, 0.2, 0.1), (3.0, 0.2, 0.2), (3.0, 0.25, 0.25),
            (3.0, 0.2, 0.3), (3.0, 0.5, 0.5), (4.0, 0.2, 0.1), (2.0, 0.2, 0.1)]


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--model", default=None, help="registry id (default: the default model)")
    ap.add_argument("--sets", nargs="+", default=["2spk", "3spk"], choices=list(SETS))
    ap.add_argument("--n", type=int, default=300, help="samples per set")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "proposed.yaml"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="JSON output (default: results/stream_eval_<id>.json)")
    args = ap.parse_args()
    setup_logging()

    reg = ModelRegistry(args.registry, resolve_device(args.device))
    be = reg.get(args.model)
    d = resolve_paths(load_config(args.config))["data"]
    results: dict = {"model": be.model_id, "n_per_set": args.n, "sets": {}}
    for set_name in args.sets:
        root = d["root"] if SETS[set_name] is None else str(PROJECT_ROOT / SETS[set_name])
        samples = val_subset(prepare_eval_set(d, root, "test"), args.n, seed=0)
        ds = TSEDataset(samples, root, None, {}, train=False, sample_rate=be.sample_rate,
                        enroll_seconds=d["enroll_seconds_eval"])
        rows: dict[str, list[float]] = {}
        compute: dict[str, list[float]] = {}
        for i in tqdm(range(len(ds)), desc=set_name):
            item = ds[i]
            mix, tgt = item["mixture"].numpy(), item["target"].numpy()
            prof = be.enroll(item["enrollment"].numpy())
            base = si_sdr_np(mix, tgt)
            rows.setdefault("offline", []).append(si_sdr_np(be.extract(mix, prof), tgt) - base)
            for w, h, la in SETTINGS:
                key = f"window {w:.0f}s hop {h * 1000:.0f} lookahead {la * 1000:.0f}"
                st = be.new_stream(prof, window=w, hop=h, lookahead=la)
                t0 = time.perf_counter()
                y = st.process_all(mix)
                compute.setdefault(key, []).append(1000 * (time.perf_counter() - t0) / max(st.calls, 1))
                rows.setdefault(key, []).append(si_sdr_np(y, tgt) - base)
        res = {}
        print(f"\n{set_name}: {len(ds)} test samples, model {be.model_id}")
        print(f"{'setting':38s} {'latency':>8s} {'SI-SDRi':>8s} {'vs offline':>10s} {'NSR':>6s} {'ms/hop':>7s}")
        off = float(np.mean(rows["offline"]))
        for key, v in rows.items():
            a = np.array(v)
            lat = "-" if key == "offline" else f"{sum(float(t) for t in key.split()[3::2]):.0f} ms"
            ms = f"{np.mean(compute[key]):.1f}" if key in compute else "-"
            print(f"{key:38s} {lat:>8s} {a.mean():8.2f} {a.mean() - off:+10.2f} {100 * (a < 0).mean():5.1f}% {ms:>7s}")
            res[key] = {"si_sdri": float(a.mean()), "nsr": float((a < 0).mean()),
                        "ms_per_hop": float(np.mean(compute[key])) if key in compute else None}
        results["sets"][set_name] = res
    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"stream_eval_{be.model_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
