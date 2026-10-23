"""Evaluate a checkpoint with fixed enrollments.

By default evaluates the test split of the training dataset (Libri2Mix) and writes
``test_metrics.json`` and ``test_per_utt.csv`` (default: next to the checkpoint).
With ``--eval-root`` any LibriMix-style dataset (Libri3Mix, our Libri4Mix, ...) is evaluated
with its own fixed enrollments; outputs are then named ``{dataset}_{split}_metrics.json``.

For mixtures with more than two speakers, ``same_gender`` means that at least one
interfering speaker has the target's gender.

Example:
    python scripts/evaluate.py --ckpt runs/proposed/best.pt
    python scripts/evaluate.py --ckpt runs/debug/best.pt --max-samples 20 --out-dir runs/debug/eval_tiny
    python scripts/evaluate.py --ckpt runs/x/best.pt --eval-root generated/Libri3Mix
    python scripts/evaluate.py --ckpt runs/x/best.pt --metric-workers 8   # parallel metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tse.config import resolve_paths  # noqa: E402
from tse.data.dataset import TSEDataset, collate_tse, prepare_eval_set, prepare_splits  # noqa: E402
from tse.engine import autocast_ctx  # noqa: E402
from tse.metrics import aggregate, available_metrics, utterance_metrics  # noqa: E402
from tse.utils import load_model_from_checkpoint, resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("evaluate")


def _init_metric_worker() -> None:
    """One thread per metric worker (the pool supplies the parallelism)."""
    torch.set_num_threads(1)


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None, help="override the dataset root stored in the ckpt")
    ap.add_argument("--split", default="test",
                    help="'test'/'val' of the training dataset, or a split of --eval-root")
    ap.add_argument("--eval-root", default=None,
                    help="evaluate another LibriMix-style dataset (e.g. generated/Libri3Mix)")
    ap.add_argument("--max-samples", type=int, default=None, help="first N samples only (debug)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-extra", action="store_true", help="skip SDR / SIR / SAR / PESQ / STOI")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", default="none", choices=["none", "bf16"])
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--metric-workers", type=int, default=0,
                    help="compute the per-utterance metrics in N processes (0: inline)")
    args = ap.parse_args()
    setup_logging()

    device = resolve_device(args.device)
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    cfg = ckpt["config"]
    if args.data_root:
        cfg = resolve_paths(cfg, args.data_root)
    dcfg = cfg["data"]
    avail = available_metrics()
    if not args.no_extra:
        for k, ok in avail.items():
            if not ok:
                log.warning("Metric %s skipped: package not installed", k.upper())

    data = prepare_splits(dcfg)
    if args.eval_root:
        eval_root = str(Path(args.eval_root).resolve())
        samples = prepare_eval_set(dcfg, eval_root, args.split)
        set_name = f"{Path(eval_root).name}_{args.split}"
    else:
        if args.split not in ("test", "val"):
            raise SystemExit("--split must be 'test' or 'val' unless --eval-root is given")
        eval_root = dcfg["root"]
        samples = data.samples[args.split]
        set_name = args.split
    if args.max_samples:
        samples = samples.iloc[:args.max_samples]
    ds = TSEDataset(samples, eval_root, None, data.spk2label, train=False,
                    sample_rate=dcfg["sample_rate"], enroll_seconds=dcfg["enroll_seconds_eval"],
                    load_interferers=not args.no_extra)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                    collate_fn=collate_tse)

    if args.metric_workers > 0:
        # one BLAS/OpenMP thread per worker (inherited by the spawned processes); otherwise each
        # worker's numpy/scipy solves spawn a full thread pool and the CPU is oversubscribed
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"
    pool = (ProcessPoolExecutor(args.metric_workers, initializer=_init_metric_worker)
            if args.metric_workers > 0 else None)
    pending = []  # (meta, metrics dict or future), in dataset order
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"eval {set_name}", mininterval=5):
            mix = batch["mixture"].to(device)
            enr = batch["enrollment"].to(device)
            with autocast_ctx(device, args.amp):
                est = model(mix, enr, compute_interference=False)["est"]
            est_np = est[0].float().cpu().numpy()
            intf = batch["interferers"][0].numpy() if "interferers" in batch else None
            margs = (est_np, batch["target"][0].numpy(), batch["mixture"][0].numpy(),
                     dcfg["sample_rate"], not args.no_extra, intf)
            m = pool.submit(utterance_metrics, *margs) if pool else utterance_metrics(*margs)
            pending.append((batch["meta"][0], m))
    rows = []
    for meta, m in tqdm(pending, desc="metrics", mininterval=5, disable=pool is None):
        m = m.result() if pool else m
        rows.append({"mixture_id": meta["mixture_id"], "target_role": meta["role"],
                     "speaker_id": meta["speaker_id"], "enrollment_path": meta["enroll_path"],
                     "length_s": meta["length"] / dcfg["sample_rate"], **m,
                     "same_gender": meta["same_gender"], "n_src": meta["n_src"]})
    if pool:
        pool.shutdown()
    elapsed = time.time() - t0

    per_utt = pd.DataFrame(rows)
    first = ["mixture_id", "target_role", "speaker_id", "enrollment_path", "length_s",
             "si_sdr_in", "si_sdr_out", "si_sdri", "same_gender"]
    per_utt = per_utt[first + [c for c in per_utt.columns if c not in first]]
    metric_rows = [{k: v for k, v in r.items() if isinstance(v, float) and k != "length_s"}
                   for r in rows]
    summary = {"checkpoint": str(Path(args.ckpt).resolve()), "dataset": str(eval_root),
               "split": args.split, "n_src": sorted({int(r["n_src"]) for r in rows}),
               "n_samples": len(rows), "eval_time_s": elapsed, "overall": aggregate(metric_rows),
               "metrics_available": avail, "notes": []}
    for k, ok in avail.items():
        if not ok or args.no_extra:
            summary["notes"].append(f"{k.upper()} not computed "
                                    f"({'disabled' if args.no_extra else 'package missing'})")
    sg = per_utt["same_gender"].astype(str)
    if (sg != "").any():
        summary["by_gender"] = {}
        for group, flag in (("same_gender", "True"), ("different_gender", "False")):
            sub = [metric_rows[i] for i in np.flatnonzero((sg == flag).to_numpy())]
            if sub:
                summary["by_gender"][group] = aggregate(sub)
    else:
        log.warning("No gender information; same/different-gender breakdown skipped")
        summary["notes"].append("gender breakdown skipped (no gender info)")

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ckpt).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    per_utt.to_csv(out_dir / f"{set_name}_per_utt.csv", index=False)
    with open(out_dir / f"{set_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    o = summary["overall"]
    print(f"\n{set_name}: {len(rows)} samples, {elapsed:.0f}s")
    for k in ("si_sdr_out", "si_sdr_in", "si_sdri", "snr_out", "snri", "sdri", "sir_out", "siri",
              "sar_out", "stoi_out", "pesq_out", "pesq_in"):
        if k in o:
            print(f"  {k:<11} mean {o[k]['mean']:7.2f}  median {o[k]['median']:7.2f}  "
                  f"std {o[k]['std']:6.2f}")
    print(f"  NSR (SI-SDRi<0): {o['nsr'] * 100:.2f}%   SI-SDRi<5dB: "
          f"{o['frac_si_sdri_below_5db'] * 100:.2f}%")
    for group, agg in summary.get("by_gender", {}).items():
        print(f"  {group:<17} SI-SDRi mean {agg['si_sdri']['mean']:6.2f} (n={agg['si_sdri']['n']}), "
              f"NSR {agg['nsr'] * 100:.2f}%")
    print(f"Wrote {out_dir / f'{set_name}_metrics.json'} and {set_name}_per_utt.csv")


if __name__ == "__main__":
    main()
