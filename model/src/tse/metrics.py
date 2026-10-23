"""Evaluation metrics: SI-SDR(i), SNR(i), SDR(i), SIR(i), SAR, PESQ, STOI and aggregation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.linalg
import scipy.signal
import torch

from .losses import si_sdr

log = logging.getLogger(__name__)

try:
    from pesq import pesq as _pesq
    HAVE_PESQ = True
except ImportError:
    HAVE_PESQ = False
try:
    from pystoi import stoi as _stoi
    HAVE_STOI = True
except ImportError:
    HAVE_STOI = False


def available_metrics() -> dict[str, bool]:
    """Which optional metrics can be computed in this environment."""
    return {"sdr_sir_sar": True, "pesq": HAVE_PESQ, "stoi": HAVE_STOI}


def si_sdr_np(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR of two 1-D numpy arrays (same formula as the training loss)."""
    return float(si_sdr(torch.from_numpy(est), torch.from_numpy(ref)))


def snr_np(est: np.ndarray, ref: np.ndarray, match_energy: bool = True,
           eps: float = 1e-8) -> float:
    """SNR in dB: ``10 log10(|ref|^2 / |ref - est|^2)``.

    The models are trained with the scale-invariant SI-SDR, so their output level is
    arbitrary; with ``match_energy`` the estimate is first rescaled to the reference energy
    (``|est| = |ref|``). Unlike SI-SDR's least-squares scaling, this keeps leakage and
    distortion energy in the estimate's level, so SNR <= SI-SDR-like values for imperfect
    estimates.
    """
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    if match_energy:
        est = est * np.sqrt((ref @ ref + eps) / (est @ est + eps))
    err = ref - est
    return float(10 * np.log10((ref @ ref + eps) / (err @ err + eps)))


def _xcorr(a: np.ndarray, b: np.ndarray, n_lags: int, nfft: int) -> np.ndarray:
    """``r[t] = sum_n a[n] b[n + t]`` for lags ``t = 0 .. n_lags-1`` (rows of a, b broadcast)."""
    fa, fb = np.fft.rfft(a, nfft), np.fft.rfft(b, nfft)
    return np.fft.irfft(np.conj(fa) * fb, nfft)[..., :n_lags]


def bss_eval_target(est: np.ndarray, ref: np.ndarray, interferers: np.ndarray,
                    filter_length: int = 512, load_diag: float = 1e-10
                    ) -> tuple[float, float, float]:
    """BSS-eval SDR, SIR and SAR of ``est`` for target ``ref`` (Vincent et al., 2006; the
    ``bss_eval_sources`` definition of mir_eval / BSS Eval v3 with 512-tap filters).

    ``interferers`` (K, T) holds each interfering source separately. The estimate is
    projected (least squares, time-invariant FIR filters) onto the target alone and onto
    all references: ``s_target = P_target est``, ``e_interf = P_all est - s_target``,
    ``e_artif = est - P_all est``; ``SDR = |s_target|^2 / |e_interf + e_artif|^2``,
    ``SIR = |s_target|^2 / |e_interf|^2`` and ``SAR = |s_target + e_interf|^2 / |e_artif|^2``.
    SDR depends only on the target projection, so it equals fast_bss_eval's single-reference
    ``sdr`` (used for the SDR numbers reported before SIR/SAR were added).

    Implemented directly in numpy/scipy (Cholesky solve of the block-Toeplitz Gram matrix):
    fast_bss_eval's numpy backend fails on numpy>=2 with several references and its torch
    backend hits an MKL ``lu_solve`` error on some inputs.
    """
    L = filter_length
    refs = np.concatenate([ref[None], interferers]).astype(np.float64)
    x = est.astype(np.float64)
    n, T = refs.shape
    nfft = int(2 ** np.ceil(np.log2(T + L)))
    # Gram matrix G[(i,a),(j,b)] = sum_t r_i[t-a] r_j[t-b] = R_ij[a-b] (zero-padded signals)
    G = np.empty((n * L, n * L))
    for i in range(n):
        for j in range(i, n):
            r_ij = _xcorr(refs[i], refs[j], L, nfft)  # sum r_i[t] r_j[t+lag], lag >= 0
            r_ji = _xcorr(refs[j], refs[i], L, nfft)
            lag = np.arange(L)[:, None] - np.arange(L)[None, :]  # a - b
            blk = np.where(lag >= 0, r_ij[np.abs(lag)], r_ji[np.abs(lag)])
            G[i * L:(i + 1) * L, j * L:(j + 1) * L] = blk
            G[j * L:(j + 1) * L, i * L:(i + 1) * L] = blk.T
    # D[(i,a)] = sum_t r_i[t-a] x[t]
    D = _xcorr(refs, x[None], L, nfft).reshape(-1)
    reg = load_diag * np.trace(G) / len(G) + 1e-12
    G[np.diag_indices_from(G)] += reg

    def project(k: int) -> np.ndarray:
        c = scipy.linalg.cho_solve(scipy.linalg.cho_factor(G[:k * L, :k * L]), D[:k * L])
        return sum(scipy.signal.fftconvolve(refs[i], c[i * L:(i + 1) * L])[:T] for i in range(k))

    s_target = project(1)
    p_all = project(n)
    e_interf = p_all - s_target
    e_artif = x - p_all
    eps = 1e-20
    sir = 10 * np.log10((s_target @ s_target + eps) / (e_interf @ e_interf + eps))
    sar = 10 * np.log10((p_all @ p_all + eps) / (e_artif @ e_artif + eps))
    e_total = x - s_target
    sdr = 10 * np.log10((s_target @ s_target + eps) / (e_total @ e_total + eps))
    return float(sdr), float(sir), float(sar)


def utterance_metrics(est: np.ndarray, ref: np.ndarray, mix: np.ndarray, sample_rate: int,
                      extra: bool = True, interferers: np.ndarray | None = None
                      ) -> dict[str, float]:
    """All metrics for one utterance. ``extra`` enables SDR/SIR/SAR/PESQ/STOI when available.

    ``interferers`` (K, T): the individual interfering sources; if None, ``mix - ref`` is used
    as a single interferer for SIR/SAR.
    """
    m: dict[str, float] = {}
    m["si_sdr_out"] = si_sdr_np(est, ref)
    m["si_sdr_in"] = si_sdr_np(mix, ref)
    m["si_sdri"] = m["si_sdr_out"] - m["si_sdr_in"]
    m["snr_out"] = snr_np(est, ref)
    m["snr_in"] = snr_np(mix, ref)
    m["snri"] = m["snr_out"] - m["snr_in"]
    if not extra:
        return m
    intf = interferers if interferers is not None else (mix - ref)[None]
    sdr_out, sir_out, sar_out = bss_eval_target(est, ref, intf)
    sdr_in, sir_in, _ = bss_eval_target(mix, ref, intf)  # mix SAR is ~inf (no artifacts)
    m.update(sdr_out=sdr_out, sdr_in=sdr_in, sdri=sdr_out - sdr_in,
             sir_out=sir_out, sir_in=sir_in, siri=sir_out - sir_in, sar_out=sar_out)
    if HAVE_PESQ and sample_rate in (8000, 16000):
        mode = "nb" if sample_rate == 8000 else "wb"
        try:
            m["pesq_out"] = float(_pesq(sample_rate, ref, est, mode))
            m["pesq_in"] = float(_pesq(sample_rate, ref, mix, mode))
        except Exception as err:  # pesq raises its own error types on silent input
            log.warning("PESQ failed: %s", err)
    if HAVE_STOI:
        m["stoi_out"] = float(_stoi(ref, est, sample_rate, extended=False))
        m["stoi_in"] = float(_stoi(ref, mix, sample_rate, extended=False))
    return m


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean / median / std of a 1-D array."""
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "n": int(len(v))}


def aggregate(per_utt: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-utterance metrics into the summary reported by ``evaluate.py``."""
    keys = [k for k in per_utt[0] if isinstance(per_utt[0][k], float)]
    out: dict[str, Any] = {k: summarize(np.array([u[k] for u in per_utt if k in u])) for k in keys}
    sdri = np.array([u["si_sdri"] for u in per_utt])
    out["nsr"] = float((sdri < 0).mean())
    out["frac_si_sdri_below_5db"] = float((sdri < 5).mean())
    return out
