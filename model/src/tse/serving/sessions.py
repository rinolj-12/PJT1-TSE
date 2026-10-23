"""Application logic shared by the demo servers (web_demo.py, realtime_server.py, serve.py).

* :class:`DemoState`: the clip mixer. It mixes 2-4 single-speaker recordings, extracts the
  target and scores the result against every source.
* :class:`Session`: one live-streaming connection. It handles the control messages (model,
  reference, start/stop, settings) and turns incoming PCM frames into extracted audio with a
  :class:`~tse.serving.streaming.ChunkedStream`.

Both talk to models only through the :class:`~tse.serving.registry.ModelRegistry`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from typing import Any

import numpy as np

from ..demo_app import SpeakerPool, mix_sources, score_extraction
from .registry import ModelRegistry
from .streaming import ChunkedStream

log = logging.getLogger(__name__)

MAX_SECONDS = 60.0      # longest clip the mixer accepts
MAX_BACKLOG_S = 1.0     # live: drop the oldest input if processing falls this far behind
STATS_EVERY_S = 0.5
HOPS = (0.1, 0.2, 0.25, 0.5)
LOOKAHEADS = (0.0, 0.1, 0.2, 0.3, 0.5)


def b64_to_f32(s: str) -> np.ndarray:
    """Decode a base64 string of little-endian float32 samples."""
    x = np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("audio contains NaN/Inf")
    return x


def f32_to_b64(x: np.ndarray) -> str:
    """Encode samples as base64 little-endian float32."""
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode("ascii")


class DemoState:
    """Models, data pool and settings shared by all requests."""

    def __init__(self, registry: ModelRegistry, pool: SpeakerPool | None) -> None:
        self.registry = registry
        self.pool = pool
        self.sample_rate = registry.get().sample_rate  # also preloads the default model
        self.rng = np.random.default_rng()

    def info(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "max_seconds": MAX_SECONDS,
            "models": [{"id": m["id"], "label": m["label"] + (" (default)" if m["default"] else "")}
                       for m in self.registry.available()],
            "random_available": self.pool is not None,
            "device": str(self.registry.device),
        }

    def random(self, n: int, gender_mode: str) -> dict[str, Any]:
        if self.pool is None:
            raise ValueError("random test speakers are unavailable (dataset not found)")
        if not 2 <= n <= 4:
            raise ValueError("n must be 2-4")
        spk = self.pool.random_speakers(n, self.rng, gender_mode)
        return {"speakers": [{
            "speaker": s["speaker"], "gender": s["gender"], "utt": s["utt"],
            "ref_utt": s["ref_utt"], "audio": f32_to_b64(s["audio"]),
            "ref_audio": f32_to_b64(s["ref_audio"]),
        } for s in spk]}

    def separate(self, req: dict[str, Any]) -> dict[str, Any]:
        sources = [b64_to_f32(s) for s in req["sources"]]
        if not 2 <= len(sources) <= 4:
            raise ValueError("give 2-4 speakers")
        target = int(req["target"])
        if not 0 <= target < len(sources):
            raise ValueError("target index out of range")
        enrollment = b64_to_f32(req["enrollment"])
        if len(enrollment) < self.sample_rate // 2:
            raise ValueError("the reference clip must be at least 0.5 s long")
        gains = [float(g) for g in req.get("gains_db", [0.0] * len(sources))]
        model_id = req.get("model") or self.registry.default_id
        if model_id not in self.registry.entries:
            raise ValueError(f"unknown model {model_id!r}")
        backend = self.registry.get(model_id)
        if backend.sample_rate != self.sample_rate:
            raise ValueError("this model uses a different sample rate than the page")
        limit = int(MAX_SECONDS * self.sample_rate)
        if max(len(s) for s in sources) > limit:
            raise ValueError(f"each recording must be at most {MAX_SECONDS:.0f} s")
        mixture, scaled = mix_sources(sources, gains, req.get("length_mode", "longest"))
        t0 = time.perf_counter()
        est = backend.extract(mixture, backend.enroll(enrollment))
        elapsed = time.perf_counter() - t0
        scores = score_extraction(est, scaled, mixture, target)
        peak = float(np.abs(est).max())
        est_play = est / peak * 0.9 if peak > 0 else est  # SI-SDR output scale is arbitrary
        log.info("separate: %d speakers, %.1f s, model %s, target %d -> SI-SDRi %.2f dB (%.0f ms)",
                 len(sources), len(mixture) / self.sample_rate, model_id, target,
                 scores["si_sdri"], elapsed * 1e3)
        return {
            "mixture": f32_to_b64(mixture),
            "estimate": f32_to_b64(est_play),
            "target": f32_to_b64(scaled[target]),
            "seconds": len(mixture) / self.sample_rate,
            "inference_ms": elapsed * 1e3,
            **scores,
        }


class Session:
    """One browser connection: chosen model, speaker profile, live stream and input backlog.

    ``ws`` is anything with ``async send(str | bytes)`` (a websockets connection or an
    adapter around a FastAPI WebSocket)."""

    def __init__(self, ws: Any, registry: ModelRegistry,
                 pool: SpeakerPool | None) -> None:
        self.ws, self.registry, self.pool = ws, registry, pool
        self.model_id = registry.default_id
        self.backend = None
        self.reference: np.ndarray | None = None
        self.profile = None
        self.stream: ChunkedStream | None = None
        self.settings = {"window": 3.0, "hop": 0.2, "lookahead": 0.1}
        self.backlog = np.zeros(0, dtype=np.float32)
        self.wake = asyncio.Event()
        self.dropped_s = 0.0
        self.rng = np.random.default_rng()

    async def send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    async def error(self, msg: str) -> None:
        await self.send({"type": "error", "message": msg})

    # ---- control messages --------------------------------------------------------------
    async def ensure_backend(self) -> None:
        if self.backend is None or self.backend.model_id != self.model_id:
            self.backend = await asyncio.to_thread(self.registry.get, self.model_id)

    async def on_hello(self, _: dict) -> None:
        await self.ensure_backend()
        await self.send({"type": "init", "models": self.registry.available(),
                         "model": self.model_id, "model_info": self.backend.info(),
                         "sample_rate": self.backend.sample_rate, "settings": self.settings,
                         "hops": HOPS, "lookaheads": LOOKAHEADS,
                         "random_available": self.pool is not None})

    async def on_set_model(self, msg: dict) -> None:
        old_sr = self.backend.sample_rate if self.backend else None
        self.model_id = msg["id"]
        t0 = time.perf_counter()
        await self.ensure_backend()
        if old_sr is not None and self.backend.sample_rate != old_sr:
            raise ValueError("this model uses a different sample rate; reload the page")
        if self.reference is not None:
            self.profile = await asyncio.to_thread(self.backend.enroll, self.reference)
            if self.stream is not None:  # hot swap, input history kept
                await asyncio.to_thread(self.stream.rebind, self.backend, self.profile)
        log.info("Switched to model %s", self.model_id)
        await self.send({"type": "model", "model": self.model_id, "model_info": self.backend.info(),
                         "switch_ms": 1000 * (time.perf_counter() - t0)})

    async def on_reference(self, msg: dict) -> None:
        ref = b64_to_f32(msg["audio"])
        await self.ensure_backend()
        self.profile = await asyncio.to_thread(self.backend.enroll, ref)
        self.reference = ref
        if self.stream is not None:
            await asyncio.to_thread(self.stream.rebind, self.backend, self.profile)
        await self.send({"type": "enrolled", "seconds": len(ref) / self.backend.sample_rate})

    def _new_stream(self) -> ChunkedStream:
        st = self.backend.new_stream(self.profile, **self.settings)
        if self.stream is not None:  # keep the input history across setting changes
            st.history = self.stream.history.copy()
        return st

    async def on_start(self, msg: dict) -> None:
        if self.profile is None:
            raise ValueError("give a reference recording first")
        self._apply_settings(msg)
        self.stream = None
        self.backlog = np.zeros(0, dtype=np.float32)
        self.dropped_s = 0.0
        self.stream = self._new_stream()
        await self.send({"type": "started", "settings": self.stream.settings()})

    def _apply_settings(self, msg: dict) -> None:
        for key, allowed in (("hop", HOPS), ("lookahead", LOOKAHEADS)):
            if key in msg:
                v = float(msg[key])
                if v not in allowed:
                    raise ValueError(f"{key} must be one of {allowed}")
                self.settings[key] = v

    async def on_settings(self, msg: dict) -> None:
        self._apply_settings(msg)
        if self.stream is not None:
            self.stream = self._new_stream()
        await self.send({"type": "settings", "settings": dict(self.settings)})

    async def on_stop(self, _: dict) -> None:
        self.stream = None
        self.backlog = np.zeros(0, dtype=np.float32)
        await self.send({"type": "stopped"})

    async def on_random_mixture(self, msg: dict) -> None:
        if self.pool is None:
            raise ValueError("random test mixtures are unavailable (dataset not found)")
        n = int(msg.get("n", 3))
        if not 2 <= n <= 4:
            raise ValueError("n must be 2-4")
        spk = await asyncio.to_thread(self.pool.random_speakers, n, self.rng,
                                      msg.get("gender", "any"))
        mix, srcs = mix_sources([s["audio"] for s in spk], [0.0] * n, "longest")
        await self.send({
            "type": "mixture", "mixture": f32_to_b64(mix), "target": f32_to_b64(srcs[0]),
            "reference": f32_to_b64(spk[0]["ref_audio"]),
            "speakers": [{"speaker": s["speaker"], "gender": s["gender"]} for s in spk],
        })

    HANDLERS = {"hello": on_hello, "set_model": on_set_model, "reference": on_reference,
                "start": on_start, "settings": on_settings, "stop": on_stop,
                "random_mixture": on_random_mixture}

    # ---- audio --------------------------------------------------------------------------
    def on_audio(self, data: bytes) -> None:
        if self.stream is None:
            return
        x = np.frombuffer(data, dtype="<f4")
        self.backlog = np.concatenate([self.backlog, x])
        limit = int(MAX_BACKLOG_S * self.stream.sr)
        if len(self.backlog) > limit:  # we are behind real time: drop the oldest input
            drop = len(self.backlog) - limit
            self.dropped_s += drop / self.stream.sr
            self.backlog = self.backlog[drop:]
        self.wake.set()

    async def worker(self) -> None:
        """Process the backlog in hop-sized steps and send the output back."""
        last_stats = 0.0
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.stream is not None and len(self.backlog) >= self.stream.hop:
                st = self.stream
                n = (len(self.backlog) // st.hop) * st.hop
                x, self.backlog = self.backlog[:n], self.backlog[n:]
                try:
                    y = await asyncio.to_thread(st.push, x)
                except (ValueError, RuntimeError) as e:
                    log.exception("streaming failed")
                    self.stream = None
                    await self.error(f"streaming stopped: {e}")
                    break
                if len(y):
                    await self.ws.send(np.ascontiguousarray(y, dtype="<f4").tobytes())
                now = time.perf_counter()
                if now - last_stats > STATS_EVERY_S:
                    last_stats = now
                    await self.send({"type": "stats", "compute_ms": st.compute_ms,
                                     "algorithmic_ms": st.latency_ms,
                                     "backlog_ms": 1000 * len(self.backlog) / st.sr,
                                     "dropped_s": self.dropped_s, "model": self.model_id})


async def dispatch(session: Session, text: str) -> None:
    """Handle one JSON control message; errors are reported to the client, not raised."""
    try:
        req = json.loads(text)
        fn = Session.HANDLERS.get(req.get("type"))
        if fn is None:
            raise ValueError(f"unknown message type {req.get('type')!r}")
        await fn(session, req)
    except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
        await session.error(str(e))
    except RuntimeError as e:  # e.g. CUDA out of memory
        log.exception("request failed")
        await session.error(str(e))
