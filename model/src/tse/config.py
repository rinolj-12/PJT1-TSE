"""YAML config loading with defaults, ``base:`` inheritance and CLI overrides.

Configs are plain nested dicts. Every key that the code reads has a default in
:data:`DEFAULTS`, so a YAML file only needs to list what it changes.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "name": "run",
    "seed": 42,
    "deterministic": False,
    "data": {
        "root": "../data/Libri2Mix",  # Libri2Mix root (contains wav8k/); override with --data-root
        "sample_rate": 8000,
        "mode": "min",  # min | max
        "mix_type": "clean",  # clean (mix_clean) | both (mix_both, noisy)
        "train_split": "train-100",
        "val_split": "dev",
        "test_split": "test",
        "cache_dir": "cache",
        "splits_dir": "splits",
        "custom_split_ratios": [0.8, 0.1, 0.1],  # only used if dev/test metadata are missing
        "segment_seconds": 3.0,
        "enroll_seconds_train": 4.0,
        "enroll_seconds_eval": 6.0,
        "enroll_seed": 1234,
        "val_subset": 500,
        "max_train_samples": None,
        "gender_file": None,  # path to LibriSpeech SPEAKERS.TXT; auto-discovered if None
        # on-the-fly 2-4 speaker mixing for training (see tse/data/dynamic_mix.py)
        "dynamic_mix": {
            "enabled": False,
            "n_spk_probs": [0.4, 0.35, 0.25],  # P(2, 3, 4 speakers incl. target)
            "level_db_range": [-5.0, 5.0],  # i.i.d. per-source level
            "partial_overlap_prob": 0.25,
            "min_overlap": 0.5,
            # P(interferers drawn from the target's top-k most similar training speakers,
            # by ECAPA speaker similarity) instead of uniformly
            "hard_pair_prob": 0.0,
            "hard_pair_top_k": 20,
        },
        # extra validation sets: [{name, root, split, subset}], e.g. Libri3Mix dev
        "extra_val_sets": [],
        "val_name": "2spk",  # name of the main (val_split) validation set in logs
    },
    "model": {
        "type": "ueconv_tse",  # ueconv_tse | tdspeakerbeam
        "N": 512,
        "L": 16,
        "stride": 8,
        "B": 128,
        "H": 512,
        "Q": 4,
        "K": 5,
        "R": 3,
        "spk_dim": 256,
        "mask_act": "relu",  # relu | sigmoid
        # proposed-model switches
        "separator": "ublock",  # ublock | tcn
        "fusion": "film",  # film | mult | concat
        "inject": "every_stage",  # every_stage | input_only
        "use_ges_interference": True,
        "use_mgfa": True,
        "mgfa_beta": "channel",  # scalar | channel
        "upsample": "nearest",  # nearest | transposed
        "se_reduction": 8,
        "attention": "se",  # U-block bottleneck: se | mhsa | diff (replaces SE)
        "attn_dim": 256,  # mhsa/diff query/key/value width
        "attn_heads": 4,  # mhsa heads; diff uses attn_heads // 2 double-width heads
        # TCN settings (baseline, and separator=tcn ablation)
        "P": 3,
        "X": 8,
        "adapt_layer": 7,
        # speaker encoder
        "spk_encoder": "resnet",  # resnet | ecapa_frozen
        "share_encoder": True,
        "spk_channels": [256, 512, 512],
    },
    "loss": {
        "lambda_ce": 0.5,
        "lambda_int": 0.1,
        "lambda_judge": 0.0,  # frozen-ECAPA "judge" loss on the output (needs dynamic mixing)
        "judge_scale": 10.0,  # cosine -> logit scale of the judge softmax
    },
    "train": {
        "batch_size": 8,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "max_epochs": 100,
        "lr_factor": 0.5,
        "lr_patience": 3,
        "early_stop_patience": 12,
        "amp": "none",  # none | bf16
        "num_workers": 4,
        "log_every": 100,
        "max_steps_per_epoch": None,
        "device": "auto",  # auto | cuda | cpu
        "init_from": None,  # checkpoint to initialise the weights from (fresh optimizer)
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base is not None:
        cfg = deep_merge(_load_yaml_with_base((path.parent / base).resolve()), cfg)
    return cfg


def set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    """Set ``cfg['a']['b'] = value`` for ``key='a.b'``; unknown keys raise ``KeyError``."""
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            raise KeyError(f"Unknown config section '{p}' in override '{key}'")
        node = node[p]
    if parts[-1] not in node:
        raise KeyError(f"Unknown config key '{key}'")
    node[parts[-1]] = value


def parse_overrides(overrides: list[str] | None) -> list[tuple[str, Any]]:
    """Parse ``["model.R=4", "train.lr=5e-4"]`` into typed ``(key, value)`` pairs."""
    pairs = []
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        k, v = item.split("=", 1)
        pairs.append((k.strip(), yaml.safe_load(v)))
    return pairs


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config on top of :data:`DEFAULTS` and apply ``key=value`` overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        cfg = deep_merge(cfg, _load_yaml_with_base(Path(path).resolve()))
    for k, v in parse_overrides(overrides):
        set_by_dotted_key(cfg, k, v)
    _validate(cfg)
    return cfg


def _validate(cfg: dict[str, Any]) -> None:
    choices = {
        ("data", "mode"): {"min", "max"},
        ("data", "mix_type"): {"clean", "both"},
        ("model", "type"): {"ueconv_tse", "tdspeakerbeam"},
        ("model", "separator"): {"ublock", "tcn"},
        ("model", "fusion"): {"film", "mult", "concat"},
        ("model", "inject"): {"every_stage", "input_only"},
        ("model", "mgfa_beta"): {"scalar", "channel"},
        ("model", "upsample"): {"nearest", "transposed"},
        ("model", "attention"): {"se", "mhsa", "diff"},
        ("model", "spk_encoder"): {"resnet", "ecapa_frozen"},
        ("model", "mask_act"): {"relu", "sigmoid"},
        ("train", "amp"): {"none", "bf16"},
    }
    for (sec, key), allowed in choices.items():
        if cfg[sec][key] not in allowed:
            raise ValueError(f"{sec}.{key}={cfg[sec][key]!r} not in {sorted(allowed)}")


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_paths(cfg: dict[str, Any], data_root: str | None = None) -> dict[str, Any]:
    """Make data paths absolute.

    ``data_root`` (from the CLI) is resolved against the current directory; relative
    paths written in the config are resolved against the project root, so commands
    work from any directory.
    """
    d = cfg["data"]
    d["root"] = str(Path(data_root).resolve()) if data_root else str(
        (PROJECT_ROOT / d["root"]).resolve())
    for key in ("cache_dir", "splits_dir"):
        d[key] = str((PROJECT_ROOT / d[key]).resolve())
    if d.get("gender_file"):
        d["gender_file"] = str((PROJECT_ROOT / d["gender_file"]).resolve())
    for vs in d.get("extra_val_sets") or []:
        vs["root"] = str((PROJECT_ROOT / vs["root"]).resolve())
    if cfg["train"].get("init_from"):
        cfg["train"]["init_from"] = str((PROJECT_ROOT / cfg["train"]["init_from"]).resolve())
    return cfg


def save_config(cfg: dict[str, Any], path: str | Path) -> None:
    """Write a config snapshot to YAML."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
