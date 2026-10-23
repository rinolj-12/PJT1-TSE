"""Sliding-window streaming for offline (non-causal) TSE models.

Our models see the whole input (global layer norm, two-sided convolutions), so live
audio is processed in overlapping windows. Every ``hop`` seconds the last ``window``
seconds of input go through the model, and the hop of output that ends ``lookahead``
seconds before the newest sample is emitted. Algorithmic latency is ``hop + lookahead``.

Two details keep the output smooth:

* **Level.** SI-SDR training leaves the output scale arbitrary, and it differs per window.
  Each window's estimate is rescaled by its least-squares projection onto the input window,
  ``g = <est, mix> / <est, est>``, which puts it at the level the target has in the
  mixture. This is absolute, so no drift accumulates.
* **Seams.** With ``lookahead > 0`` the previous window already estimated the start of the
  next hop. The two estimates are linearly crossfaded over ``crossfade`` seconds.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from .backends import SpeakerProfile, TSEBackend


class ChunkedStream:
    """Stateful live extraction: feed input with :meth:`push`, get output samples back.

    Output comes in whole hops: after ``n`` hops of input, ``n * hop`` output samples have
    been returned, and output sample ``m`` corresponds to input sample ``m - lookahead``.
    So a sample is heard ``lookahead`` plus up to ``hop`` seconds after it was spoken.

    Args:
        backend: The model; uses :meth:`TSEBackend.extract_window`.
        profile: Target-speaker profile from ``backend.enroll``.
        window: Model input length in seconds (training used 3 s segments).
        hop: Seconds of new output per model call.
        lookahead: Future context (seconds) after the emitted hop.
        crossfade: Seam crossfade in seconds (at most ``lookahead``).
        min_context: Below this much input the stream outputs silence.
    """

    def __init__(self, backend: TSEBackend, profile: SpeakerProfile, window: float = 3.0,
                 hop: float = 0.2, lookahead: float = 0.1, crossfade: float = 0.02,
                 min_context: float = 0.5) -> None:
        sr = backend.sample_rate
        self.backend, self.profile, self.sr = backend, profile, sr
        self.hop = max(1, int(round(hop * sr)))
        self.lookahead = max(0, int(round(lookahead * sr)))
        self.window = max(int(round(window * sr)), self.hop + self.lookahead)
        self.xf = min(int(round(crossfade * sr)), self.lookahead)
        self.min_context = int(min_context * sr)
        self.history = np.zeros(0, dtype=np.float32)   # last `window` input samples
        self.pending = np.zeros(0, dtype=np.float32)   # input not yet consumed (< hop)
        self.tail = np.zeros(0, dtype=np.float32)      # estimate of the next hop's first xf samples
        self.n_in = self.n_out = 0
        self._lock = threading.Lock()  # push() may run in a worker thread while rebind() swaps
        self.calls = 0
        self.compute_ms = 0.0

    @property
    def latency_samples(self) -> int:
        """Algorithmic latency (hop + lookahead), excluding compute and audio I/O."""
        return self.hop + self.lookahead

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sr

    def settings(self) -> dict[str, Any]:
        return {"window": self.window / self.sr, "hop": self.hop / self.sr,
                "lookahead": self.lookahead / self.sr, "crossfade": self.xf / self.sr,
                "latency_ms": self.latency_ms}

    def rebind(self, backend: TSEBackend, profile: SpeakerProfile) -> "ChunkedStream":
        """Switch model mid-stream, keeping the input history so there is no gap."""
        if backend.sample_rate != self.sr:
            raise ValueError("cannot hot-swap to a model with a different sample rate")
        with self._lock:
            self.backend, self.profile = backend, profile
            self.tail = np.zeros(0, dtype=np.float32)  # the old model's estimate: no crossfade
        return self

    def push(self, chunk: np.ndarray) -> np.ndarray:
        """Consume input samples; return the output produced (a multiple of ``hop``)."""
        chunk = np.asarray(chunk, dtype=np.float32)
        with self._lock:
            self.n_in += len(chunk)
            self.pending = np.concatenate([self.pending, chunk])
            produced = []
            while len(self.pending) >= self.hop:
                new, self.pending = self.pending[:self.hop], self.pending[self.hop:]
                self.history = np.concatenate([self.history, new])[-self.window:]
                produced.append(self._step())
        out = np.concatenate(produced) if produced else np.zeros(0, dtype=np.float32)
        self.n_out += len(out)
        return out

    def process_all(self, x: np.ndarray, chunk: int = 160) -> np.ndarray:
        """Stream a whole signal through in ``chunk``-sample pieces (as live audio would
        arrive) and return the output aligned with the input (same length)."""
        n = len(x)
        outs = [self.push(x[i:i + chunk]) for i in range(0, n, chunk)]
        tail = n + self.lookahead + self.hop - self.n_in  # flush with silence
        outs.append(self.push(np.zeros(max(tail, 0), dtype=np.float32)))
        y = np.concatenate(outs)
        return y[self.lookahead:self.lookahead + n]

    def _step(self) -> np.ndarray:
        """Run the model on the current window; return the next ``hop`` output samples."""
        h, la, xf = self.hop, self.lookahead, self.xf
        if len(self.history) < max(self.min_context, h + la):
            self.tail = np.zeros(0, dtype=np.float32)
            return np.zeros(h, dtype=np.float32)
        x = self.history
        t0 = time.perf_counter()
        est = self.backend.extract_window(x, self.profile).astype(np.float32)
        self.compute_ms = 1000.0 * (time.perf_counter() - t0)
        self.calls += 1
        energy = float(est @ est)
        g = float(est @ x) / energy if energy > 1e-12 else 0.0
        est = est * g
        end = len(x) - la                       # emitted hop is est[end - h : end]
        seg = est[end - h:end].copy()
        if xf and len(self.tail) == xf:
            fade = np.linspace(0.0, 1.0, xf, dtype=np.float32)
            seg[:xf] = self.tail * (1 - fade) + seg[:xf] * fade
        self.tail = est[end:end + xf].copy() if xf else np.zeros(0, dtype=np.float32)
        return seg
