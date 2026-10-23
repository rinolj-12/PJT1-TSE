"""Edge-deployment benchmark: complexity, ONNX export, quantisation accuracy and CPU latency.

For each checkpoint the model is split into the two graphs an edge device runs:

* ``embedder``  enrollment (1, Te) -> speaker embedding (1, D): run **once** per target speaker;
* ``extractor`` mixture (1, T) + embedding (1, D) -> target estimate (1, T): run per audio
  window (the interference branch is training-only and is not exported).

Parts (``--parts``, default all):

``complexity``  deployed parameters (ONNX initialisers), GFLOPs of the extractor per second of
                audio and of the embedder per 4 s enrollment, model file sizes (fp32 / int8).
``export``      ONNX export (opset 17, dynamic length) of both graphs, ONNX Runtime (ORT) vs
                PyTorch check at a length different from the export length, and ORT dynamic
                INT8 quantisation (per-channel weights).
``accuracy``    SI-SDRi / NSR on fixed, seeded subsets of the 2-, 3- and 4-speaker test sets
                (fixed enrollments) for: PyTorch fp32 (reference), PyTorch fp16 (GPU),
                ORT fp32 and ORT int8 (CPU). Also the SI-SDR of each variant's output
                against the fp32 output (how much the deployed model deviates).
``latency``     CPU wall time of one 3 s streaming window (the unit the real-time streamer
                processes per hop) and the real-time factor, for PyTorch fp32 and ORT
                fp32/int8 at 1, 4 and 8 threads; the process runs at high priority.

Outputs: ``<out>/<model>/{embedder,extractor}{,_int8}.onnx`` and ``<out>/edge_benchmark.json``.
Latencies on a desktop CPU are an upper bound on edge throughput, not edge numbers.

Example:
    python scripts/edge_benchmark.py --models r6=saved_models/multispk_r6_ep79/best.pt \\
        r3=runs/finetune_multispk/best.pt --n 150
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402
from tqdm import tqdm  # noqa: E402

from tse.config import PROJECT_ROOT  # noqa: E402
from tse.data.dataset import TSEDataset, prepare_eval_set, prepare_splits, val_subset  # noqa: E402
from tse.metrics import si_sdr_np  # noqa: E402
from tse.utils import load_model_from_checkpoint, setup_logging  # noqa: E402

log = logging.getLogger("edge_benchmark")
SR = 8000
SETS = {"2spk": None, "3spk": "generated/Libri3Mix", "4spk": "generated/Libri4Mix"}


class Embedder(nn.Module):
    """enrollment (1, Te) -> embedding (1, D)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, enrollment: torch.Tensor) -> torch.Tensor:
        return self.model.embed(enrollment)


class Extractor(nn.Module):
    """mixture (1, T) + embedding (1, D) -> estimate (1, T)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, mixture: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return self.model(mixture, None, compute_interference=False, embedding=embedding)["est"]


def gflops(fn, *inputs) -> float:
    """Total GFLOPs (multiply-add = 2 FLOPs) of one call."""
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        fn(*inputs)
    return fc.get_total_flops() / 1e9


def onnx_params(path: Path) -> int:
    """Number of parameters stored as initialisers in an ONNX file."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    return int(sum(numpy_helper.to_array(t).size for t in m.graph.initializer))


def ort_session(path: Path, threads: int):
    """ONNX Runtime CPU session with ``threads`` intra-op threads."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def export(model: nn.Module, out_dir: Path, spk_dim: int) -> dict:
    """Export embedder/extractor to ONNX, verify against PyTorch, quantise to INT8."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_m, ext_m = Embedder(model).eval(), Extractor(model).eval()
    enr, mix = torch.randn(1, 4 * SR), torch.randn(1, 3 * SR)
    with torch.no_grad():
        e = emb_m(enr)
    paths = {"embedder": out_dir / "embedder.onnx", "extractor": out_dir / "extractor.onnx"}
    torch.onnx.export(emb_m, (enr,), str(paths["embedder"]), input_names=["enrollment"],
                      output_names=["embedding"], dynamic_axes={"enrollment": {1: "Te"}},
                      opset_version=17, dynamo=False)
    torch.onnx.export(ext_m, (mix, e), str(paths["extractor"]),
                      input_names=["mixture", "embedding"], output_names=["estimate"],
                      dynamic_axes={"mixture": {1: "T"}, "estimate": {1: "T"}},
                      opset_version=17, dynamo=False)
    res: dict = {}
    for name, p in list(paths.items()):
        q = out_dir / f"{name}_int8.onnx"
        quantize_dynamic(str(p), str(q), weight_type=QuantType.QUInt8, per_channel=True)
        paths[f"{name}_int8"] = q
    # numerical check at lengths different from the export lengths
    enr2, mix2 = torch.randn(1, 5 * SR + 123), torch.randn(1, 4 * SR + 77)
    with torch.no_grad():
        e_t = emb_m(enr2)
        y_t = ext_m(mix2, e_t)
    e_o = ort_session(paths["embedder"], 4).run(None, {"enrollment": enr2.numpy()})[0]
    y_o = ort_session(paths["extractor"], 4).run(
        None, {"mixture": mix2.numpy(), "embedding": e_t.numpy()})[0]
    res["check"] = {
        "embedding_max_abs_diff": float(np.abs(e_o - e_t.numpy()).max()),
        "estimate_max_abs_diff": float(np.abs(y_o - y_t.numpy()).max()),
        "estimate_si_sdr_vs_torch_db": si_sdr_np(y_o[0], y_t[0].numpy()),
        "estimate_length_ok": bool(y_o.shape == tuple(y_t.shape)),
    }
    res["files_MB"] = {k: p.stat().st_size / 2**20 for k, p in paths.items()}
    res["params_deployed"] = {k: onnx_params(paths[k]) for k in ("embedder", "extractor")}
    res["paths"] = {k: str(p.relative_to(PROJECT_ROOT)) for k, p in paths.items()}
    return res


def complexity(model: nn.Module) -> dict:
    """Parameter counts and GFLOPs of the deployed graphs."""
    emb_m, ext_m = Embedder(model).eval(), Extractor(model).eval()
    with torch.no_grad():
        e = emb_m(torch.randn(1, 4 * SR))
    return {
        "params_total_M": sum(p.numel() for p in model.parameters()) / 1e6,
        "gflops_extractor_per_s_audio": gflops(ext_m, torch.randn(1, 4 * SR), e) / 4,
        "gflops_extractor_3s_window": gflops(ext_m, torch.randn(1, 3 * SR), e),
        "gflops_embedder_4s_enrollment": gflops(emb_m, torch.randn(1, 4 * SR)),
    }


def eval_samples(ckpt: dict, n: int) -> dict[str, TSEDataset]:
    """Fixed, seeded test subsets (same mixtures and enrollments for every variant)."""
    d = ckpt["config"]["data"]
    data = prepare_splits(d)
    out = {}
    for name, root in SETS.items():
        samples = data.samples["test"] if root is None else prepare_eval_set(d, root, "test")
        out[name] = TSEDataset(val_subset(samples, n, seed=0), d["root"] if root is None else root,
                               None, {}, train=False, sample_rate=SR,
                               enroll_seconds=d["enroll_seconds_eval"])
    return out


def accuracy(model: nn.Module, paths: dict, sets: dict[str, TSEDataset], threads: int) -> dict:
    """SI-SDRi / NSR per variant and set, plus deviation from the fp32 output."""
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    emb_m, ext_m = Embedder(model).eval(), Extractor(model).eval()
    m16 = None
    if gpu is not None:
        import copy
        m16 = copy.deepcopy(model).to(gpu).half().eval()
    sess = {k: ort_session(Path(PROJECT_ROOT / paths[k]), threads)
            for k in ("embedder", "extractor", "embedder_int8", "extractor_int8")}
    res: dict = {}
    for set_name, ds in sets.items():
        acc: dict[str, list[float]] = {}
        dev: dict[str, list[float]] = {}
        for i in tqdm(range(len(ds)), desc=f"accuracy {set_name}", mininterval=10):
            it = ds[i]
            mix, tgt, enr = it["mixture"][None], it["target"].numpy(), it["enrollment"][None]
            base = si_sdr_np(mix[0].numpy(), tgt)
            with torch.no_grad():
                y32 = ext_m(mix, emb_m(enr))[0].numpy()
                outs = {"torch_fp32": y32}
                if m16 is not None:
                    e16 = m16.embed(enr.to(gpu).half())
                    outs["torch_fp16"] = m16(mix.to(gpu).half(), None, compute_interference=False,
                                             embedding=e16)["est"][0].float().cpu().numpy()
            for tag, sfx in (("ort_fp32", ""), ("ort_int8", "_int8")):
                e = sess["embedder" + sfx].run(None, {"enrollment": enr.numpy()})[0]
                outs[tag] = sess["extractor" + sfx].run(
                    None, {"mixture": mix.numpy(), "embedding": e})[0][0]
            for tag, y in outs.items():
                acc.setdefault(tag, []).append(si_sdr_np(y, tgt) - base)
                if tag != "torch_fp32":
                    dev.setdefault(tag, []).append(si_sdr_np(y, y32))
        ref = float(np.mean(acc["torch_fp32"]))
        res[set_name] = {tag: {"si_sdri": float(np.mean(v)), "delta_vs_fp32": float(np.mean(v)) - ref,
                               "nsr": float((np.array(v) < 0).mean()),
                               "si_sdr_vs_fp32_output_db": (float(np.median(dev[tag]))
                                                            if tag in dev else None),
                               "n": len(v)}
                         for tag, v in acc.items()}
    return res


def timeit(fn, warmup: int = 3, reps: int = 15) -> dict:
    """Median / p90 wall time in ms."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t0))
    ts.sort()
    return {"median_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * (len(ts) - 1))]}


def latency(model: nn.Module, paths: dict, threads_list: list[int], window_s: float) -> dict:
    """CPU latency of one streaming window, embedder latency, and real-time factors."""
    try:
        import psutil
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -10)
    except Exception as err:  # priority is best effort
        log.warning("could not raise priority: %s", err)
    emb_m, ext_m = Embedder(model).eval(), Extractor(model).eval()
    mix = torch.randn(1, int(window_s * SR))
    enr = torch.randn(1, 4 * SR)
    with torch.no_grad():
        e = emb_m(enr)
    res: dict = {}
    for th in threads_list:
        torch.set_num_threads(th)
        r: dict = {}
        with torch.no_grad():
            r["torch_fp32"] = timeit(lambda: ext_m(mix, e))
            r["torch_fp32_embedder_4s"] = timeit(lambda: emb_m(enr), reps=8)
        for tag, sfx in (("ort_fp32", ""), ("ort_int8", "_int8")):
            s_ext = ort_session(Path(PROJECT_ROOT / paths["extractor" + sfx]), th)
            s_emb = ort_session(Path(PROJECT_ROOT / paths["embedder" + sfx]), th)
            feed = {"mixture": mix.numpy(), "embedding": e.numpy()}
            r[tag] = timeit(lambda: s_ext.run(None, feed))
            r[tag + "_embedder_4s"] = timeit(lambda: s_emb.run(None, {"enrollment": enr.numpy()}),
                                             reps=8)
        for k in list(r):
            if "embedder" not in k:
                r[k]["rtf"] = r[k]["median_ms"] / (1000 * window_s)
        res[f"{th}_threads"] = r
        log.info("%d threads: %s", th, {k: round(v["median_ms"], 1) for k, v in r.items()})
    torch.set_num_threads(os.cpu_count() or 1)
    return res


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint pairs")
    ap.add_argument("--parts", nargs="+", default=["complexity", "export", "accuracy", "latency"])
    ap.add_argument("--n", type=int, default=150, help="test samples per set (accuracy)")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--window", type=float, default=3.0, help="streaming window (s) for latency")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "results" / "edge"))
    args = ap.parse_args()
    setup_logging()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jpath = out / "edge_benchmark.json"
    results = json.loads(jpath.read_text()) if jpath.is_file() else {}
    results["host"] = {"cpu": platform.processor(), "cpu_count": os.cpu_count(),
                       "torch": torch.__version__,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    for spec in args.models:
        name, ck = spec.split("=", 1)
        model, ckpt = load_model_from_checkpoint(ck, torch.device("cpu"))
        r = results.setdefault(name, {"checkpoint": ck})
        if "complexity" in args.parts:
            r["complexity"] = complexity(model)
            log.info("%s complexity: %s", name, r["complexity"])
        if "export" in args.parts:
            r["export"] = export(model, out / name, ckpt["config"]["model"]["spk_dim"])
            log.info("%s export: %s", name, r["export"]["check"])
        if "accuracy" in args.parts:
            r["accuracy"] = accuracy(model, r["export"]["paths"], eval_samples(ckpt, args.n),
                                     threads=8)
            for s, v in r["accuracy"].items():
                log.info("%s %s: %s", name, s, {k: round(x["si_sdri"], 2) for k, x in v.items()})
        if "latency" in args.parts:
            r["latency"] = latency(model, r["export"]["paths"], args.threads, args.window)
        jpath.write_text(json.dumps(results, indent=2))
    print(f"Wrote {jpath}")


if __name__ == "__main__":
    main()
