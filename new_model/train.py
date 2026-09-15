import argparse
import json
import shutil
import time
import zipfile
import numpy as np
import torch
from pathlib import Path
from dataclasses import asdict

from model import (
    Config, set_seed, setup_dirs, load_and_prepare_data, train_hastnet,
    collect_test_stats, bootstrap_rmse, const_velocity_stats, per_horizon_rmse_correct,
    HASTNet, make_loader, HASTDataset, OUT_DIR, CKPT_DIR, FIG_DIR, TUNE_DIR, WORK,
    official_rmse
)
from visualization import plot_baseline, plot_tuning


def run_tune(base_cfg, prep, idx_df, device, n_trials):
    """Randomized Search over hyperparameter space."""
    param_space = {
        "d_model": [128, 192, 256],
        "dropout": (0.05, 0.30),  # uniform
        "lr": [1e-4, 3e-4, 5e-4, 1e-3],
        "weight_decay": [0.01, 0.05, 0.1],
        "huber_delta": [0.2, 0.35, 0.5],
        "w_end": (0.1, 1.5),  # uniform
        "n_control": [8, 12, 16],
    }

    rng = np.random.default_rng(base_cfg.seed)
    tune_results = []

    print("[tuning] Training baseline for comparison...")
    cfg_base = Config(**asdict(base_cfg))
    cfg_base.exp_name = "tune_baseline"
    cfg_base.epochs = 12
    cfg_base.patience = 4
    model_b, ema_b, hist_b, _ = train_hastnet(cfg_base, prep, idx_df, device, TUNE_DIR / "outputs",
                                              TUNE_DIR / "checkpoints")
    base_best_val = min(hist_b["val_rmse"])
    n_base = sum(p.numel() for p in model_b.parameters())

    for i in range(n_trials):
        overrides = {}
        for k, v in param_space.items():
            if isinstance(v, tuple):
                overrides[k] = float(rng.uniform(v[0], v[1]))
            else:
                overrides[k] = rng.choice(v)

        overrides["d_model"] = int(overrides["d_model"])
        overrides["n_control"] = int(overrides["n_control"])

        cfg_t = Config(**{**asdict(base_cfg), **overrides})
        cfg_t.exp_name = f"tune_{i}"
        cfg_t.epochs = 12
        cfg_t.patience = 4

        t0 = time.time()
        model_t, ema_t, hist_t, _ = train_hastnet(cfg_t, prep, idx_df, device, TUNE_DIR / "outputs",
                                                  TUNE_DIR / "checkpoints")
        dt = time.time() - t0

        best_ep = int(np.argmin(hist_t["val_rmse"]))
        best_val = float(min(hist_t["val_rmse"]))

        tune_results.append(dict(
            name=cfg_t.exp_name, overrides=overrides,
            n_params=int(sum(p.numel() for p in model_t.parameters())),
            best_epoch=best_ep, best_val_rmse=best_val,
            final_val_rmse=float(hist_t["val_rmse"][-1]),
            time_sec=round(dt, 1), history=hist_t
        ))
        print(f"[tuning] {cfg_t.exp_name}: best val RMSE {best_val:.4f} @ epoch {best_ep} ({dt / 60:.1f} min)")

    best = min(tune_results, key=lambda r: r["best_val_rmse"])

    print(f"[tuning] Evaluating best config '{best['name']}' on test set...")
    cfg_best = Config(**{**asdict(base_cfg), **best["overrides"]})
    cfg_best.exp_name = best["name"]
    te_ds = HASTDataset(idx_df, cfg_best, cfg_best.test_weeks, prep, train=False)
    te_loader = make_loader(te_ds, False, cfg_best)
    model_best = HASTNet(cfg_best).to(device)
    ck = torch.load(TUNE_DIR / "checkpoints" / f"{best['name']}_best.pt", map_location=device)
    model_best.load_state_dict(ck["ema"])
    model_best.eval()

    stats_b = collect_test_stats(model_best, te_loader, cfg_best, device)
    ptb, lob, hib = bootstrap_rmse(stats_b["plays_sq"], stats_b["plays_n"], rng, cfg_best.n_bootstrap,
                                   cfg_best.bootstrap_block)

    plot_tuning(hist_b, tune_results, n_base, TUNE_DIR / "figures")

    tuning_summary = dict(
        baseline=dict(best_val_rmse=base_best_val, n_params=n_base),
        configs=[{k: r[k] for k in ("name", "overrides", "n_params", "best_epoch", "best_val_rmse", "time_sec")} for r
                 in tune_results],
        best_config=best["name"], best_test_rmse=ptb, best_test_ci=[lob, hib]
    )
    with open(TUNE_DIR / "outputs" / "tuning_results.json", "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print(f"[tuning] Best config: {best['name']} with test RMSE {ptb:.4f} [{lob:.4f}, {hib:.4f}]")


def main():
    parser = argparse.ArgumentParser(description="HAST-Net Training Pipeline")
    parser.add_argument("--mode", choices=["train", "tune", "smoke_test"], default="train",
                        help="Execution mode: standard training, hyperparameter tuning (Randomized Search), or quick smoke test.")
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of trials for Randomized Search in 'tune' mode.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    args = parser.parse_args()

    setup_dirs()
    cfg = Config()
    cfg.seed = args.seed
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    is_smoke = (args.mode == "smoke_test")
    if is_smoke:
        print("--- RUNNING IN SMOKE TEST MODE ---")
        cfg.epochs = 1
        cfg.train_weeks = [1]
        cfg.val_weeks = [1]
        cfg.test_weeks = [1]
        cfg.patience = 2
        cfg.exp_name = "smoke_test"
        cfg.n_bootstrap = 10  # reduce bootstrap for speed

    prep, idx_df = load_and_prepare_data(cfg, smoke_test=is_smoke)

    if args.mode == "train" or args.mode == "smoke_test":
        model, ema, history, te_loader = train_hastnet(cfg, prep, idx_df, device, OUT_DIR, CKPT_DIR)

        if not is_smoke:
            print("Evaluating on test set...")
            rng = np.random.default_rng(cfg.seed)
            stats = collect_test_stats(model, te_loader, cfg, device)
            pt, lo, hi = bootstrap_rmse(stats["plays_sq"], stats["plays_n"], rng, cfg.n_bootstrap, cfg.bootstrap_block)
            bsq, bn = const_velocity_stats(te_loader, cfg)
            bpt, blo, bhi = bootstrap_rmse(bsq, bn, rng, cfg.n_bootstrap, cfg.bootstrap_block)

            results = {
                "HAST-Net (EMA)": dict(rmse=pt, ci_lo=lo, ci_hi=hi),
                "constant-velocity": dict(rmse=bpt, ci_lo=blo, ci_hi=bhi)
            }

            hor_mean, hor_lo, hor_hi = per_horizon_rmse_correct(model, te_loader, device, seed=cfg.seed)
            plot_baseline(history, results, hor_mean, hor_lo, hor_hi, stats, model, te_loader, device, FIG_DIR)

            with open(OUT_DIR / "test_results.json", "w") as f:
                json.dump(dict(results=results, horizon_rmse=hor_mean.tolist(),
                               horizon_ci=[hor_lo.tolist(), hor_hi.tolist()]), f, indent=2)
            print(f"Test RMSE: {pt:.4f} [{lo:.4f}, {hi:.4f}]")
        else:
            print("--- SMOKE TEST PASSED SUCCESSFULLY! NO ERRORS DETECTED. ---")

    elif args.mode == "tune":
        print("--- RUNNING HYPERPARAMETER TUNING (RANDOMIZED SEARCH) ---")
        run_tune(cfg, prep, idx_df, device, args.n_trials)


if __name__ == "__main__":
    main()