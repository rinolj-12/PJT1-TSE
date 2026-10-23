"""Extract a target speaker from one mixture given an enrollment recording.

Example:
    python scripts/infer.py --ckpt runs/proposed/best.pt --mix mix.wav --enroll enroll.wav --out out.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio.functional as AF  # noqa: E402

from tse.utils import load_model_from_checkpoint, resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("infer")


def load_mono(path: str, target_sr: int) -> torch.Tensor:
    """Load audio as mono float32 at ``target_sr`` (resampling if needed). Returns (T,)."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(x.mean(axis=1))
    if sr != target_sr:
        log.info("Resampling %s from %d to %d Hz", path, sr, target_sr)
        wav = AF.resample(wav, sr, target_sr)
    return wav


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mix", required=True)
    ap.add_argument("--enroll", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-enroll-seconds", type=float, default=None,
                    help="center-crop the enrollment (default: config enroll_seconds_eval)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    setup_logging()

    device = resolve_device(args.device)
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    sr = ckpt["config"]["data"]["sample_rate"]
    mix = load_mono(args.mix, sr)
    enr = load_mono(args.enroll, sr)
    max_e = args.max_enroll_seconds or ckpt["config"]["data"]["enroll_seconds_eval"]
    n_e = int(max_e * sr)
    if len(enr) > n_e:
        s = (len(enr) - n_e) // 2
        enr = enr[s:s + n_e]
    with torch.no_grad():
        est = model(mix[None].to(device), enr[None].to(device), compute_interference=False)["est"]
    out = est[0].cpu().numpy()
    peak = np.abs(out).max()
    if peak > 1.0:
        out = out / peak * 0.99
        log.info("Output peak %.2f > 1; normalised to avoid clipping", peak)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out, sr)
    print(f"Wrote {args.out}: {len(out) / sr:.2f}s at {sr} Hz (input {len(mix) / sr:.2f}s)")


if __name__ == "__main__":
    main()
