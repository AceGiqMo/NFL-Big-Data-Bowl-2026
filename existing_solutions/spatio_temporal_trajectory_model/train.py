"""
train.py — entry point: training, evaluation, baselines, EDA, multi-seed summary.

Interactions
    * imports model.py        (config, data, architecture, optimizers, metrics)
    * imports visualization.py (all figures)
Commands
    python train.py eda
    python train.py baselines
    python train.py train    [--seed S] [--epochs N] [--batch-size B] [--weeks 1,2]
    python train.py evaluate [--exp-name NAME]
    python train.py summarize --exp-names seed_42,seed_43,seed_44

Bug-fix history (ported from the Kaggle training session)
    FIX-4  optimizers handled as a list (1 or 2 of them).
    FIX-5  AMP opt-in; loss always computed in float32 under autocast.
    FIX-6  .cpu() before .numpy() for every GPU tensor in evaluation.
    FIX-7  fallback checkpoint save when val metric is NaN/inf or no best yet
           (prevents FileNotFoundError in `evaluate` after a diverged run).
    FIX-8  explicit warning when the training loss becomes NaN.
"""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

import visualization as viz
from model import (FIG_ROOT, OUT_ROOT, Config, EMA, NFLDataset, ROLE_LIST,
                   TrajectoryModel, build_optimizers, ckpt_path,
                   deltas_to_absolute, exp_dirs, huber_delta_loss,
                   load_all_plays, make_loader, official_rmse, set_seed)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def to_device(batch: Dict, device: torch.device) -> Dict:
    """Move every tensor of a collated batch to the target device."""
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


@contextmanager
def ema_weights(model: nn.Module, ema: EMA):
    """Temporarily swap in EMA weights for evaluation, then restore live ones."""
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    try:
        yield
    finally:
        model.load_state_dict(live)


def lr_lambda_factory(total_steps: int, warmup: int):
    """Linear warmup followed by cosine decay to zero."""
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(prog, 1.0))))
    return fn


# FIX-5: AMP compatibility helpers (work across torch 2.x versions)
if hasattr(torch.amp, "GradScaler"):
    def make_scaler(enabled: bool):
        return torch.amp.GradScaler("cuda", enabled=enabled)
else:  # older torch
    def make_scaler(enabled: bool):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def make_autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.float16, enabled=enabled)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss_rmse(model, loader, cfg, device) -> Tuple[float, float]:
    """Validation pass with EMA-friendly caller; returns (huber, official RMSE)."""
    model.eval()
    tot_loss = tot_sq = tot_n = 0.0
    ac = make_autocast(cfg.amp and device.type == "cuda")
    for batch in loader:
        batch = to_device(batch, device)
        with ac:
            pred = model(batch)
            mask = batch["valid"] * batch["to_pred"][..., None]
            loss = huber_delta_loss(pred.float(), batch["dY"], mask, cfg.huber_delta)
        tot_loss += loss.item() * mask.sum().item()
        pred_abs = deltas_to_absolute(pred.float().cpu().numpy(), batch["last_xy"].cpu().numpy())
        true_abs = deltas_to_absolute(batch["dY"].cpu().numpy(), batch["last_xy"].cpu().numpy())
        m = mask.cpu().numpy() > 0
        tot_sq += float(((pred_abs - true_abs) ** 2).sum(-1)[m].sum())
        tot_n += int(m.sum())
    return tot_loss / max(1, tot_n), official_rmse(tot_sq, tot_n)


def train(cfg: Config) -> dict:
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    out_dir, fig_dir = exp_dirs(cfg.exp_name)

    plays = load_all_plays(cfg)
    cache: Dict = {}
    train_ds = NFLDataset(plays, cfg, cfg.train_weeks, cache)
    val_ds = NFLDataset(plays, cfg, cfg.val_weeks, cache)
    train_loader = make_loader(train_ds, cfg, True)
    val_loader = make_loader(val_ds, cfg, False)

    model = TrajectoryModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[info] exp={cfg.exp_name} device={device} params={n_params:,} "
          f"train={len(train_ds)} val={len(val_ds)}")

    opts = build_optimizers(model, cfg)                    # FIX-4: list of optimizers
    total_steps = max(1, len(train_loader) * cfg.epochs)
    lam = lr_lambda_factory(total_steps, cfg.warmup_steps)
    scheds = [LambdaLR(opt, lam) for opt in opts]
    ema = EMA(model, cfg.ema_decay)
    use_amp = cfg.amp and device.type == "cuda"            # FIX-5
    scaler = make_scaler(use_amp)
    ac = make_autocast(use_amp)

    history = dict(train_loss=[], val_loss=[], val_rmse=[],
                   best_epoch=-1, best_val_rmse=math.inf)
    bad = 0
    for epoch in range(cfg.epochs):
        model.train()
        loss_sum = mask_sum = 0.0
        pbar = tqdm(train_loader, desc=f"train ep{epoch:02d}", leave=False)
        for batch in pbar:
            batch = to_device(batch, device)
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            with ac:
                pred = model(batch)
                mask = batch["valid"] * batch["to_pred"][..., None]
                loss = huber_delta_loss(pred.float(), batch["dY"], mask, cfg.huber_delta)
            scaler.scale(loss).backward()
            for opt in opts:                               # unscale + clip each group
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]],
                                         cfg.grad_clip)
            for opt in opts:
                scaler.step(opt)
            scaler.update()
            for s in scheds:
                s.step()
            ema.update(model)
            n = mask.sum().item()
            loss_sum += loss.item() * n
            mask_sum += n
            pbar.set_postfix(loss=loss.item(), lr=opts[-1].param_groups[0]["lr"])

        current_loss = loss_sum / max(1.0, mask_sum)
        if math.isnan(current_loss):                       # FIX-8
            print(f"[WARNING] Training loss became NaN at epoch {epoch}!")
        history["train_loss"].append(current_loss)

        with ema_weights(model, ema):
            val_loss, val_rmse = evaluate_loss_rmse(model, val_loader, cfg, device)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        tqdm.write(f"epoch {epoch:02d} | train_loss={current_loss:.4f} "
                   f"val_loss={val_loss:.4f} val_rmse={val_rmse:.4f}")

        if val_rmse < history["best_val_rmse"] - 1e-4:
            history.update(best_val_rmse=val_rmse, best_epoch=epoch)
            bad = 0
            torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                            cfg=asdict(cfg), epoch=epoch, val_rmse=val_rmse),
                       ckpt_path(cfg.exp_name))
        else:
            bad += 1
            # FIX-7: fallback save so `evaluate` never hits FileNotFoundError,
            # even if the run diverged (NaN metric) or never improved.
            if math.isnan(val_rmse) or math.isinf(val_rmse) or history["best_epoch"] == -1:
                torch.save(dict(model=model.state_dict(), ema=ema.state_dict(),
                                cfg=asdict(cfg), epoch=epoch, val_rmse=val_rmse),
                           ckpt_path(cfg.exp_name))
            if bad >= cfg.patience:
                tqdm.write(f"[info] early stopping at epoch {epoch} "
                           f"(best_epoch={history['best_epoch']})")
                break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    viz.plot_training_history(history, fig_dir / "training_history.png")
    return history


# --------------------------------------------------------------------------- #
# Test statistics (model + baseline), bootstrap CIs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_test_statistics(model, loader, cfg, device, n_samples: int = 4) -> dict:
    """Per-play squared errors, horizon/role breakdowns, radial errors, samples."""
    model.eval()
    plays_err, hor_h, hor_e, radial = [], [], [], []
    role_rmse: Dict[int, List[float]] = {}
    samples: List[dict] = []
    ac = make_autocast(cfg.amp and device.type == "cuda")
    for batch in tqdm(loader, desc="test eval", leave=False):
        batch = to_device(batch, device)
        with ac:
            pred = model(batch).float().cpu().numpy()
        pred_abs = deltas_to_absolute(pred, batch["last_xy"].cpu().numpy())
        true_abs = deltas_to_absolute(batch["dY"].cpu().numpy(), batch["last_xy"].cpu().numpy())
        mask = (batch["valid"].cpu().numpy() > 0) & (batch["to_pred"].cpu().numpy() > 0)[..., None]
        sq = ((pred_abs - true_abs) ** 2).sum(-1)                      # (B, P, O)
        radial.append(np.sqrt(sq[mask]))
        for b in range(sq.shape[0]):
            Pi, Oi = int(batch["p_mask"][b].sum()), int(batch["o_mask"][b, 0].sum())
            m, sq_b = mask[b, :Pi, :Oi], sq[b, :Pi, :Oi]
            plays_err.append(sq_b[m])
            hh = np.arange(Oi)[None, :].repeat(Pi, 0)
            hor_h.append(hh[m])
            hor_e.append(sq_b[m])
            roles = batch["roles"][b, :Pi].cpu().numpy()               # FIX-6
            for p in range(Pi):
                if m[p].any():
                    role_rmse.setdefault(int(roles[p]), []).append(
                        float(np.sqrt(sq_b[p][m[p]].mean() / 2)))
            if len(samples) < n_samples:
                samples.append(dict(
                    play=f"{int(batch['S'][b])}in/{Oi}out",
                    land=batch["land"][b].cpu().numpy(),               # FIX-6
                    players=[(ROLE_LIST[int(roles[p])], true_abs[b, p, :Oi],
                              pred_abs[b, p, :Oi],
                              batch["last_xy"][b, p].cpu().numpy())    # FIX-6
                             for p in range(Pi) if m[p].any()]))
    return dict(plays_err=plays_err,
                horizon=(np.concatenate(hor_h), np.concatenate(hor_e)),
                radial=np.concatenate(radial),
                role_rmse=role_rmse, samples=samples)


@torch.no_grad()
def constant_velocity_baseline(loader, device) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Baseline: future deltas = last observed pre-pass delta (per player)."""
    errs, hors = [], []
    for batch in tqdm(loader, desc="baseline cv", leave=False):
        batch = to_device(batch, device)                               # FIX-6
        O = int(batch["dY"].shape[2])
        pred = batch["dX"][:, :, -1:].expand(-1, -1, O, -1)
        pred_abs = deltas_to_absolute(pred.cpu().numpy(), batch["last_xy"].cpu().numpy())
        true_abs = deltas_to_absolute(batch["dY"].cpu().numpy(), batch["last_xy"].cpu().numpy())
        mask = (batch["valid"].cpu().numpy() > 0) & (batch["to_pred"].cpu().numpy() > 0)[..., None]
        sq = ((pred_abs - true_abs) ** 2).sum(-1)
        for b in range(sq.shape[0]):
            Pi, Oi = int(batch["p_mask"][b].sum()), int(batch["o_mask"][b, 0].sum())
            m = mask[b, :Pi, :Oi]
            errs.append(sq[b, :Pi, :Oi][m])
            hors.append(np.arange(Oi)[None, :].repeat(Pi, 0)[m])
    return errs, hors


def bootstrap_rmse_ci(plays_err: List[np.ndarray], rng: np.random.Generator,
                      n_boot: int, block: int) -> Tuple[float, float, float]:
    """Block bootstrap over plays (temporal dependence within a week)."""
    point = official_rmse(float(sum(e.sum() for e in plays_err)),
                          int(sum(e.size for e in plays_err)))
    n = len(plays_err)
    if n == 0:
        return point, point, point
    stats = []
    for _ in range(n_boot):
        sampled, taken = [], 0
        while taken < n:
            start = int(rng.integers(0, n))
            sampled.extend(plays_err[start:start + block])
            taken += block
        stats.append(official_rmse(float(sum(e.sum() for e in sampled)),
                                   int(sum(e.size for e in sampled))))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #
def cmd_train(args) -> None:
    cfg = Config()
    if args.weeks:
        cfg.train_weeks = [int(w) for w in args.weeks.split(",")]
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.seed is not None:
        cfg.seed = args.seed
    if args.exp_name:
        cfg.exp_name = args.exp_name
    train(cfg)


def cmd_evaluate(args) -> None:
    cfg = Config()
    if args.exp_name:
        cfg.exp_name = args.exp_name
    set_seed(cfg.seed)                     # deterministic bootstrap / loader order
    device = cfg.resolve_device()
    out_dir, fig_dir = exp_dirs(cfg.exp_name)

    plays = load_all_plays(cfg)
    cache: Dict = {}
    test_loader = make_loader(NFLDataset(plays, cfg, cfg.test_weeks, cache), cfg, False)

    ckpt = torch.load(ckpt_path(cfg.exp_name), map_location="cpu", weights_only=False)
    model = TrajectoryModel(cfg)
    model.load_state_dict(ckpt["ema"])     # evaluate EMA weights
    model.to(device)

    stats = collect_test_statistics(model, test_loader, cfg, device)
    point, lo, hi = bootstrap_rmse_ci(stats["plays_err"], np.random.default_rng(cfg.seed),
                                      cfg.n_bootstrap, cfg.bootstrap_block)
    results = {"proposed (EMA)": dict(rmse=point, ci_lo=lo, ci_hi=hi)}

    cv_err, cv_hor = constant_velocity_baseline(test_loader, device)
    p2, l2, h2 = bootstrap_rmse_ci(cv_err, np.random.default_rng(cfg.seed),
                                   cfg.n_bootstrap, cfg.bootstrap_block)
    results["constant-velocity"] = dict(rmse=p2, ci_lo=l2, ci_hi=h2)

    viz.plot_rmse_comparison(results, fig_dir / "rmse_comparison.png")
    viz.plot_rmse_vs_horizon({"proposed": stats["horizon"],
                              "constant-velocity": (np.concatenate(cv_hor),
                                                    np.concatenate(cv_err))},
                             fig_dir / "rmse_vs_horizon.png")
    viz.plot_role_breakdown({"proposed": {ROLE_LIST[r]: np.array(v)
                                          for r, v in stats["role_rmse"].items()}},
                            ROLE_LIST, fig_dir / "role_breakdown.png")
    viz.plot_trajectories(stats["samples"], fig_dir / "trajectories.png")
    viz.plot_error_qq(stats["radial"], fig_dir / "error_qq.png")

    report = dict(exp=cfg.exp_name, n_test_plays=len(test_loader.dataset),
                  results=results,
                  role_rmse={"proposed": {ROLE_LIST[r]: float(np.mean(v))
                                          for r, v in stats["role_rmse"].items()}},
                  best_epoch=ckpt["epoch"], checkpoint=str(ckpt_path(cfg.exp_name)))
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def cmd_baselines(args) -> None:
    cfg = Config()
    set_seed(cfg.seed)
    device = cfg.resolve_device()
    plays = load_all_plays(cfg)
    cache: Dict = {}
    loader = make_loader(NFLDataset(plays, cfg, cfg.test_weeks, cache), cfg, False)
    errs, _ = constant_velocity_baseline(loader, device)
    point, lo, hi = bootstrap_rmse_ci(errs, np.random.default_rng(cfg.seed),
                                      cfg.n_bootstrap, cfg.bootstrap_block)
    out_dir, _ = exp_dirs("baseline_cv")
    payload = dict(baseline="constant-velocity", rmse=point, ci_lo=lo, ci_hi=hi)
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2))


def cmd_eda(args) -> None:
    """Quick dataset statistics: frames per play, players, scored players."""
    cfg = Config()
    plays = load_all_plays(cfg)
    rows = []
    for _, row in tqdm(plays.iterrows(), total=len(plays), desc="eda"):
        rows.append(dict(week=row.week, S=int(row.inp.frame_id.max()),
                         O=int(row.out.frame_id.max()),
                         P=int(row.inp.nfl_id.nunique()),
                         scored=int(row.inp.player_to_predict.sum())))
    df = pd.DataFrame(rows)
    print(df.describe())
    viz.plot_eda(df, FIG_ROOT / "eda.png")


def cmd_summarize(args) -> None:
    """Aggregate several single-seed runs: mean +/- std (statistical basis)."""
    names = [n for n in args.exp_names.split(",") if n]
    values = {}
    for name in names:
        with open(exp_dirs(name)[0] / "test_results.json") as f:
            values[name] = json.load(f)["results"]["proposed (EMA)"]["rmse"]
    arr = np.array(list(values.values()))
    summary = dict(exps=names, rmse_per_seed=arr.tolist(),
                   mean=float(arr.mean()),
                   std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)
    print(json.dumps(summary, indent=2))
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    viz.plot_seed_summary(values, FIG_ROOT / "seed_summary.png")
    with open(OUT_ROOT / "seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="NFL trajectory prediction project")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="train the model")
    p.add_argument("--weeks", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--exp-name", type=str, default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="test metrics + figures for a checkpoint")
    p.add_argument("--exp-name", type=str, default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("baselines", help="constant-velocity baseline on test split")
    p.set_defaults(func=cmd_baselines)

    p = sub.add_parser("eda", help="dataset statistics")
    p.set_defaults(func=cmd_eda)

    p = sub.add_parser("summarize", help="mean+/-std across seed runs")
    p.add_argument("--exp-names", type=str, required=True)
    p.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()