"""Profile parameters, GFLOPs, peak GPU memory and latency of the TSE models.

Inference setting: 4 s mixture at 8 kHz, batch 1, interference branch off. The speaker
encoder runs on a 4 s enrollment and is included in the totals (separate columns show the
separator-only cost, i.e. with a cached speaker embedding).

Example:
    python scripts/profile_model.py --configs configs/proposed.yaml configs/baseline_tdspeakerbeam.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from tse.config import PROJECT_ROOT, load_config  # noqa: E402
from tse.models import build_model  # noqa: E402
from tse.utils import count_parameters, resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("profile")


def count_flops(fn) -> int:
    """Total FLOPs of ``fn()`` counted by ``FlopCounterMode``."""
    with FlopCounterMode(display=False) as fc:
        fn()
    return fc.get_total_flops()


def time_fn(fn, device: torch.device, runs: int, warmup: int) -> float:
    """Median wall-clock latency (ms) of ``fn()`` over ``runs`` runs after ``warmup``."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


@torch.no_grad()
def profile(cfg_path: str, seconds: float, device: torch.device, runs: int, warmup: int,
            n_speakers: int) -> dict:
    """Profile one config."""
    cfg = load_config(cfg_path)
    sr = cfg["data"]["sample_rate"]
    model = build_model(cfg["model"], n_speakers, sr).to(device).eval()
    T = int(seconds * sr)
    mix = torch.randn(1, T, device=device)
    enr = torch.randn(1, T, device=device)
    n_classifier = sum(p.numel() for p in model.classifier.parameters())

    def full() -> torch.Tensor:
        return model(mix, enr, compute_interference=False)["est"]

    def spk_only() -> torch.Tensor:
        return model.embed(enr)

    res = {
        "config": cfg_path, "model": cfg["model"]["type"],
        "params_M": count_parameters(model) / 1e6,
        "params_inference_M": (count_parameters(model) - n_classifier) / 1e6,
        "gflops_total": count_flops(full) / 1e9,
        "gflops_spk_encoder": count_flops(spk_only) / 1e9,
    }
    res["gflops_separator"] = res["gflops_total"] - res["gflops_spk_encoder"]
    if cfg["model"]["type"] == "ueconv_tse" and cfg["model"]["use_ges_interference"]:
        model.train()  # count the training-time forward with the interference branch
        res["gflops_train_forward_with_interf"] = count_flops(
            lambda: model(mix, enr, compute_interference=True)) / 1e9
        model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        full()
        torch.cuda.synchronize()
        res["peak_gpu_mem_MB"] = (torch.cuda.max_memory_allocated(device) - base) / 2**20
        res["weights_MB"] = base / 2**20
    res["latency_ms"] = time_fn(full, device, runs, warmup)
    res["rtf"] = res["latency_ms"] / 1000 / seconds
    return res


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--configs", nargs="+", default=["configs/proposed.yaml",
                                                     "configs/baseline_tdspeakerbeam.yaml"])
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n-speakers", type=int, default=251, help="classifier size (train only)")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "profile.json"))
    args = ap.parse_args()
    setup_logging()
    device = resolve_device(args.device)
    torch.backends.cudnn.benchmark = True
    results = [profile(c, args.seconds, device, args.runs, args.warmup, args.n_speakers)
               for c in args.configs]

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"\nInput: {args.seconds:.0f} s mixture + {args.seconds:.0f} s enrollment, 8 kHz, "
          f"batch 1, device {dev_name}, fp32")
    hdr = (f"{'model':<15}{'params(M)':>10}{'GFLOPs':>9}{'sep GFLOPs':>11}{'spk GFLOPs':>11}"
           f"{'peak MB':>9}{'lat ms':>8}{'RTF':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r['model']:<15}{r['params_M']:>10.2f}{r['gflops_total']:>9.2f}"
              f"{r['gflops_separator']:>11.2f}{r['gflops_spk_encoder']:>11.2f}"
              f"{r.get('peak_gpu_mem_MB', float('nan')):>9.1f}{r['latency_ms']:>8.2f}"
              f"{r['rtf']:>8.4f}")
    Path(args.out).write_text(json.dumps({"device": dev_name, "seconds": args.seconds,
                                          "results": results}, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
