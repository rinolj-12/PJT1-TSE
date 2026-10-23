"""Model registry: ``models.yaml`` lists the models the demos can use and names the default.

Entries look like::

    default: r6_multispk
    models:
      - id: r6_multispk
        label: "R=6, trained on 2-4 speakers"
        backend: ueconv                     # see tse.serving.backends.register_backend
        checkpoint: saved_models/multispk_r6_ep79/best.pt
        metrics_dir: runs/multispk_deep_r6  # optional; where *_test_metrics.json live
        notes: "..."

Relative paths are resolved against the project root. Backends are loaded lazily and
cached (least recently used are evicted beyond ``max_loaded``). Swapping the model a demo
uses means changing ``default`` (``scripts/register_model.py --make-default``) or picking
another id at runtime; no code changes are needed.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import yaml

from ..config import PROJECT_ROOT
from .backends import TSEBackend, backend_class

log = logging.getLogger(__name__)

DEFAULT_REGISTRY = PROJECT_ROOT / "models.yaml"
METRIC_FILES = [("2spk", "test_metrics.json"), ("3spk", "Libri3Mix_test_metrics.json"),
                ("4spk", "Libri4Mix_test_metrics.json")]
HEADER = ("# Models available to the demos (scripts/web_demo.py, scripts/realtime_server.py).\n"
          "# Edit by hand or with scripts/register_model.py; `default` is the model used\n"
          "# unless another one is picked. Relative paths are relative to the project root.\n")


def _resolve(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_metrics(entry: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Test-set summary (SI-SDRi mean, NSR, n) per speaker count, from ``metrics_dir`` (or
    the checkpoint's folder). Missing files are skipped."""
    d = _resolve(entry.get("metrics_dir")) or _resolve(entry["checkpoint"]).parent
    out = {}
    for name, fname in METRIC_FILES:
        f = d / fname
        if not f.is_file():
            continue
        m = json.loads(f.read_text(encoding="utf-8"))
        s = m.get("overall", m)
        try:
            out[name] = {"si_sdri": float(s["si_sdri"]["mean"]), "nsr": float(s["nsr"]),
                         "n": int(m.get("n_samples", 0))}
        except (KeyError, TypeError, ValueError):
            log.warning("Unrecognised metrics format in %s", f)
    return out


class ModelRegistry:
    """Thread-safe view of ``models.yaml`` with lazily loaded backends."""

    def __init__(self, path: str | Path = DEFAULT_REGISTRY, device: torch.device | None = None,
                 max_loaded: int = 4) -> None:
        self.path = Path(path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_loaded = max_loaded
        self._loaded: OrderedDict[str, TSEBackend] = OrderedDict()
        self._lock = threading.RLock()
        self.reload()

    # ---- the file -------------------------------------------------------------------
    def reload(self) -> None:
        """Re-read ``models.yaml`` (cached backends of unchanged entries are kept)."""
        with self._lock:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            entries = data.get("models") or []
            ids = [e["id"] for e in entries]
            if len(set(ids)) != len(ids):
                raise ValueError(f"duplicate model ids in {self.path}")
            for e in entries:
                for key in ("id", "label", "backend", "checkpoint"):
                    if key not in e:
                        raise ValueError(f"{self.path}: model entry missing {key!r}: {e}")
            self.entries = {e["id"]: e for e in entries}
            self.default_id = data.get("default") or (ids[0] if ids else None)
            if self.default_id not in self.entries:
                raise ValueError(f"{self.path}: default {self.default_id!r} is not a listed model")
            for mid in [m for m in self._loaded if m not in self.entries]:
                del self._loaded[mid]

    def save(self) -> None:
        with self._lock:
            body = yaml.safe_dump({"default": self.default_id, "models": list(self.entries.values())},
                                  sort_keys=False, allow_unicode=True)
            self.path.write_text(HEADER + body, encoding="utf-8")

    def add(self, entry: dict[str, Any], make_default: bool = False, replace: bool = False) -> None:
        """Add (or with ``replace``, overwrite) an entry and save the file."""
        with self._lock:
            for key in ("id", "label", "backend", "checkpoint"):
                if key not in entry:
                    raise ValueError(f"entry missing {key!r}")
            backend_class(entry["backend"])  # validates the backend name
            if not _resolve(entry["checkpoint"]).is_file():
                raise FileNotFoundError(entry["checkpoint"])
            if entry["id"] in self.entries and not replace:
                raise ValueError(f"model id {entry['id']!r} exists (use replace)")
            self.entries[entry["id"]] = entry
            self._loaded.pop(entry["id"], None)
            if make_default:
                self.default_id = entry["id"]
            self.save()

    def set_default(self, model_id: str) -> None:
        with self._lock:
            if model_id not in self.entries:
                raise KeyError(model_id)
            self.default_id = model_id
            self.save()

    # ---- listing --------------------------------------------------------------------
    def available(self) -> list[dict[str, Any]]:
        """Entries whose checkpoint exists, with their test metrics, default first."""
        out = []
        for mid, e in self.entries.items():
            if not _resolve(e["checkpoint"]).is_file():
                continue
            out.append({"id": mid, "label": e["label"], "backend": e["backend"],
                        "default": mid == self.default_id, "notes": e.get("notes", ""),
                        "metrics": read_metrics(e), "loaded": mid in self._loaded})
        out.sort(key=lambda m: not m["default"])
        return out

    # ---- backends -------------------------------------------------------------------
    def get(self, model_id: str | None = None) -> TSEBackend:
        """The backend for ``model_id`` (default model if None), loading it if needed."""
        with self._lock:
            mid = model_id or self.default_id
            if mid not in self.entries:
                raise KeyError(f"unknown model {mid!r}")
            if mid in self._loaded:
                self._loaded.move_to_end(mid)
                return self._loaded[mid]
            e = self.entries[mid]
            log.info("Loading model %s (%s)", mid, e["checkpoint"])
            cls = backend_class(e["backend"])
            extra = {k: v for k, v in e.items() if k not in ("id", "label", "backend", "checkpoint")}
            be = cls(mid, e["label"], self.device, checkpoint=_resolve(e["checkpoint"]), **extra)
            self._loaded[mid] = be
            while len(self._loaded) > self.max_loaded:
                old, _ = self._loaded.popitem(last=False)
                log.info("Unloaded model %s", old)
            return be
