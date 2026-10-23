"""Train a TSE model.

Example:
    python scripts/train.py --config configs/proposed.yaml
    python scripts/train.py --config configs/proposed.yaml --resume
    python scripts/train.py --config configs/proposed.yaml --name abl_nomgfa --set model.use_mgfa=false
    python scripts/train.py --config configs/finetune_multispk.yaml     # 2-4 speaker fine-tuning

With ``data.extra_val_sets`` the model is validated on several sets every epoch (e.g. 2-, 3-
and 4-speaker dev); checkpoint selection, LR schedule and early stopping then use the mean
SI-SDRi over all sets.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths, save_config  # noqa: E402
from tse.data.dataset import (TSEDataset, collate_tse, prepare_eval_set, prepare_splits,  # noqa: E402
                              val_subset)
from tse.data.dynamic_mix import DynamicMixDataset  # noqa: E402
from tse.engine import train_one_epoch, validate  # noqa: E402
from tse.losses import TSELoss  # noqa: E402
from tse.models.speaker_encoder import FrozenECAPA  # noqa: E402
from tse.speaker_similarity import load_or_compute_speaker_embeddings, most_similar  # noqa: E402
from tse.models import build_model  # noqa: E402
from tse.utils import (count_parameters, load_checkpoint, resolve_device, save_checkpoint,  # noqa: E402
                       seed_everything, setup_logging)

log = logging.getLogger("train")

LOG_FIELDS = ["epoch", "lr", "train_loss", "train_si_sdr", "train_ce", "train_int_si_sdr",
              "val_si_sdr", "val_si_sdri", "val_nsr", "skipped_steps", "epoch_time_s",
              "val_time_s", "best_val_si_sdri"]


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--name", default=None, help="run name (default: config 'name')")
    ap.add_argument("--runs-dir", default=str(PROJECT_ROOT / "runs"))
    ap.add_argument("--resume", action="store_true", help="resume from runs/<name>/last.pt")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()

    cfg = resolve_paths(load_config(args.config, args.set), args.data_root)
    if args.name:
        cfg["name"] = args.name
    run_dir = Path(args.runs_dir) / cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if last_path.exists() and not args.resume:
        raise SystemExit(f"{last_path} exists; use --resume or a different --name")

    seed_everything(cfg["seed"], cfg["deterministic"])
    device = resolve_device(cfg["train"]["device"])
    tcfg, dcfg = cfg["train"], cfg["data"]
    if tcfg["amp"] == "bf16" and device.type == "cpu":
        log.warning("bf16 autocast on CPU is slow; disabling")
        tcfg["amp"] = "none"
    save_config(cfg, run_dir / "config.yaml")

    data = prepare_splits(dcfg)
    train_samples = data.samples["train"]
    if dcfg["max_train_samples"]:
        rng = np.random.default_rng(cfg["seed"])
        keep = rng.choice(len(train_samples), dcfg["max_train_samples"], replace=False)
        train_samples = train_samples.iloc[np.sort(keep)]
    sr = dcfg["sample_rate"]
    dm = dcfg["dynamic_mix"]
    common = dict(sample_rate=sr, segment_seconds=dcfg["segment_seconds"],
                  enroll_seconds=dcfg["enroll_seconds_train"])
    if dm["enabled"]:
        similar = None
        if dm["hard_pair_prob"] > 0:
            emb = load_or_compute_speaker_embeddings(
                Path(dcfg["cache_dir"]) / f"ecapa_spk_{dcfg['train_split']}.pt",
                data.pools["train"], dcfg["root"], device, sample_rate=dcfg["sample_rate"])
            similar = most_similar(emb, dm["hard_pair_top_k"])
        train_ds = DynamicMixDataset(train_samples, dcfg["root"], data.pools["train"],
                                     data.spk2label, n_spk_probs=dm["n_spk_probs"],
                                     level_db_range=tuple(dm["level_db_range"]),
                                     partial_overlap_prob=dm["partial_overlap_prob"],
                                     min_overlap=dm["min_overlap"], similar_speakers=similar,
                                     hard_pair_prob=dm["hard_pair_prob"], **common)
        log.info("Dynamic mixing: P(2,3,4.. speakers)=%s, levels U%s dB, partial overlap p=%.2f, "
                 "hard pairs p=%.2f (top %d)", dm["n_spk_probs"], dm["level_db_range"],
                 dm["partial_overlap_prob"], dm["hard_pair_prob"], dm["hard_pair_top_k"])
    else:
        train_ds = TSEDataset(train_samples, dcfg["root"], data.pools["train"], data.spk2label,
                              train=True, **common)

    # validation sets: the main val split + optional extra sets (e.g. 3-/4-speaker dev)
    val_sets = [(dcfg["val_name"], dcfg["root"],
                 val_subset(data.samples["val"], dcfg["val_subset"], seed=0))]
    for vs in dcfg["extra_val_sets"]:
        samples = prepare_eval_set(dcfg, vs["root"], vs["split"])
        val_sets.append((vs["name"], vs["root"], val_subset(samples, vs.get("subset"), seed=0)))
    nw = tcfg["num_workers"]
    g = torch.Generator().manual_seed(cfg["seed"])
    train_dl = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=nw, collate_fn=collate_tse, generator=g,
                          pin_memory=device.type == "cuda", persistent_workers=nw > 0)
    val_dls = []
    for name, root, samples in val_sets:
        ds = TSEDataset(samples, root, None, data.spk2label, train=False, sample_rate=sr,
                        enroll_seconds=dcfg["enroll_seconds_eval"])
        val_dls.append((name, DataLoader(ds, batch_size=1, shuffle=False,
                                         num_workers=min(nw, 2), collate_fn=collate_tse,
                                         persistent_workers=nw > 0)))
    log.info("train samples %d (%d steps/epoch), val sets %s, train speakers %d",
             len(train_ds), len(train_dl), {n: len(d.dataset) for n, d in val_dls},
             len(data.spk2label))

    model = build_model(cfg["model"], len(data.spk2label), sr).to(device)
    log.info("Model %s: %.2fM parameters (%.2fM trainable)", cfg["model"]["type"],
             count_parameters(model) / 1e6, count_parameters(model, True) / 1e6)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=tcfg["lr_factor"], patience=tcfg["lr_patience"])
    lcfg = cfg["loss"]
    judge = None
    if lcfg["lambda_judge"] > 0:
        if not dm["enabled"]:
            raise ValueError("loss.lambda_judge > 0 needs data.dynamic_mix (individual interferers)")
        judge = FrozenECAPA(sr).to(device)
        log.info("Judge loss: frozen ECAPA on the output, weight %.2f, scale %.1f",
                 lcfg["lambda_judge"], lcfg["judge_scale"])
    loss_fn = TSELoss(lcfg["lambda_ce"], lcfg["lambda_int"], lcfg["lambda_judge"], judge,
                      lcfg["judge_scale"])
    if tcfg["init_from"] and not (args.resume and last_path.exists()):
        init = load_checkpoint(tcfg["init_from"], device)
        model.load_state_dict(init["model"])
        log.info("Initialised weights from %s (epoch %s); optimizer starts fresh with lr %.1e",
                 tcfg["init_from"], init["state"].get("epoch"), tcfg["lr"])

    multi = len(val_dls) > 1
    log_fields = list(LOG_FIELDS)
    if judge is not None:
        log_fields += ["train_judge", "train_judge_acc"]
    if multi:
        log_fields += [f"val_{n}_{m}" for n, _ in val_dls for m in ("si_sdr", "si_sdri", "nsr")]

    start_epoch, global_step, best, bad_epochs = 1, 0, -float("inf"), 0
    if args.resume and last_path.exists():
        ck = load_checkpoint(last_path, device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        st = ck["state"]
        start_epoch, global_step = st["epoch"] + 1, st["global_step"]
        best, bad_epochs = st["best"], st["bad_epochs"]
        if "rng_torch" in st:
            torch.set_rng_state(st["rng_torch"].cpu())  # map_location may move it to GPU
        log.info("Resumed from epoch %d (best val SI-SDRi %.2f)", st["epoch"], best)
    g.manual_seed(cfg["seed"] + start_epoch)  # different shuffling after a resume

    writer = SummaryWriter(str(run_dir / "tb"))
    log_csv = run_dir / "train_log.csv"
    new_csv = not log_csv.exists()
    with open(log_csv, "a", newline="", encoding="utf-8") as fcsv:
        csv_w = csv.DictWriter(fcsv, fieldnames=log_fields)
        if new_csv:
            csv_w.writeheader()
        for epoch in range(start_epoch, tcfg["max_epochs"] + 1):
            lr = optimizer.param_groups[0]["lr"]
            tr, global_step = train_one_epoch(model, train_dl, loss_fn, optimizer, device, epoch,
                                              tcfg, writer, global_step)
            t0 = time.time()
            per_set = {n: validate(model, dl, device, tcfg["amp"]) for n, dl in val_dls}
            # selection metric: mean over validation sets (= the main set if there is one)
            va = {m: float(np.mean([v[m] for v in per_set.values()]))
                  for m in ("si_sdr", "si_sdri", "nsr")}
            val_time = time.time() - t0
            scheduler.step(va["si_sdri"])
            improved = va["si_sdri"] > best
            if improved:
                best, bad_epochs = va["si_sdri"], 0
            else:
                bad_epochs += 1
            state = {"epoch": epoch, "global_step": global_step, "best": best,
                     "bad_epochs": bad_epochs, "val": va, "val_sets": per_set,
                     "rng_torch": torch.get_rng_state()}
            save_checkpoint(last_path, model, cfg, data.spk2label, optimizer, scheduler, state)
            if improved:
                save_checkpoint(best_path, model, cfg, data.spk2label, state=state)

            row = {"epoch": epoch, "lr": lr, "train_loss": tr.get("total", float("nan")),
                   "train_si_sdr": tr.get("si_sdr", float("nan")), "train_ce": tr.get("ce", 0.0),
                   "train_int_si_sdr": tr.get("int_si_sdr", 0.0), "val_si_sdr": va["si_sdr"],
                   "val_si_sdri": va["si_sdri"], "val_nsr": va["nsr"],
                   "skipped_steps": tr["skipped_steps"], "epoch_time_s": tr["epoch_time_s"],
                   "val_time_s": val_time, "best_val_si_sdri": best}
            if judge is not None:
                row.update(train_judge=tr.get("judge", float("nan")),
                           train_judge_acc=tr.get("judge_acc", float("nan")))
            if multi:
                for n, v in per_set.items():
                    row.update({f"val_{n}_{m}": v[m] for m in ("si_sdr", "si_sdri", "nsr")})
                    writer.add_scalar(f"epoch/val_{n}_si_sdri", v["si_sdri"], epoch)
                    writer.add_scalar(f"epoch/val_{n}_nsr", v["nsr"], epoch)
            csv_w.writerow(row)
            fcsv.flush()
            for k in ("train_loss", "train_si_sdr", "train_ce", "val_si_sdr", "val_si_sdri",
                      "val_nsr", "lr"):
                writer.add_scalar(f"epoch/{k}", row[k], epoch)
            log.info("epoch %d | train loss %.3f si_sdr %.2f | val si_sdr %.2f si_sdri %.2f "
                     "nsr %.3f | lr %.2e | %.0fs+%.0fs%s", epoch, row["train_loss"],
                     row["train_si_sdr"], va["si_sdr"], va["si_sdri"], va["nsr"], lr,
                     tr["epoch_time_s"], val_time, "  *best*" if improved else "")
            if multi:
                log.info("         per set: %s", "  ".join(
                    f"{n}: si_sdri {v['si_sdri']:.2f} nsr {v['nsr']:.3f}" for n, v in per_set.items()))
            if bad_epochs >= tcfg["early_stop_patience"]:
                log.info("Early stopping: no improvement for %d epochs", bad_epochs)
                break
    writer.close()
    log.info("Done. Best val SI-SDRi %.2f dB. Checkpoints in %s", best, run_dir)


if __name__ == "__main__":
    main()
