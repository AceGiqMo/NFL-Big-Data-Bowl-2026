import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path


def plot_baseline(history, results, hor_mean, hor_lo, hor_hi, stats, model, te_loader, device, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3})

    # (1) training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train loss")
    ax[0].plot(history["val_rmse"], label="val RMSE (EMA)")
    ax[0].set_xlabel("epoch");
    ax[0].legend();
    ax[0].set_title("Training / validation")
    parts = pd.DataFrame(history["parts"])
    parts.plot(ax=ax[1], title="loss components")
    ax[1].set_xlabel("epoch")
    fig.tight_layout();
    fig.savefig(fig_dir / "training_curves.png");
    plt.close(fig)

    # (2) RMSE comparison
    names = list(results);
    pts = np.array([results[n]["rmse"] for n in names])
    los = np.array([results[n]["ci_lo"] for n in names]);
    his = np.array([results[n]["ci_hi"] for n in names])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(names, pts, yerr=[pts - los, his - pts], capsize=6, color=["#4c72b0", "#dd8452"])
    ax.set_ylabel("test RMSE (yards)");
    ax.set_title("Test RMSE with 95% CI")
    fig.tight_layout();
    fig.savefig(fig_dir / "rmse_comparison.png");
    plt.close(fig)

    # (3) RMSE vs horizon
    xs = np.arange(len(hor_mean))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, hor_mean, marker="o", label="HAST-Net")
    ax.fill_between(xs, hor_lo, hor_hi, alpha=.2)
    ax.set_xlabel("future frame o");
    ax.set_ylabel("RMSE(o) (yards)")
    ax.set_title("Error growth over horizon");
    ax.legend()
    fig.tight_layout();
    fig.savefig(fig_dir / "rmse_vs_horizon.png");
    plt.close(fig)

    # (4) error percentiles
    obs = stats["sig_obs"]
    qs = np.linspace(1, 99, 99)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(qs, np.percentile(obs, qs), marker=".", ls="")
    ax.set_xlabel("percentile");
    ax.set_ylabel("radial error (yards)")
    ax.set_title("Error percentiles");
    fig.tight_layout()
    fig.savefig(fig_dir / "error_percentiles.png");
    plt.close(fig)

    # (5) sigma calibration
    sp, so = stats["sig_pred"], stats["sig_obs"]
    bins = np.quantile(sp, np.linspace(0, 1, 11))
    bx = [0.5 * (bins[i] + bins[i + 1]) for i in range(10)]
    by = [so[(sp >= bins[i]) & (sp < bins[i + 1])].mean() for i in range(10)]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(bx, by, "o-", label="observed mean |err|")
    ax.plot([0, max(bx)], [0, max(bx)], "k--", label="ideal")
    ax.set_xlabel("predicted sigma");
    ax.set_ylabel("observed error")
    ax.set_title("Sigma calibration");
    ax.legend()
    fig.tight_layout();
    fig.savefig(fig_dir / "sigma_calibration.png");
    plt.close(fig)

    # (6) trajectories
    @torch.no_grad()
    def traj_examples(model, loader, device, n=4):
        from model import predict_pos
        model.eval();
        b = next(iter(loader))
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        _, pos = predict_pos(model, b)
        pos = pos.cpu().numpy();
        true = b["pos"].cpu().numpy()
        m = (b["valid"] * b["to_pred"][..., None]).bool().cpu().numpy()
        fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.3), squeeze=False)
        for ax, i in zip(axes[0], range(n)):
            Pi = int(b["p_mask"][i].sum());
            Oi = int(b["o_mask"][i, 0].sum())
            for p in range(Pi):
                if m[i, p, 0]:
                    ax.plot(true[i, p, :Oi, 0], true[i, p, :Oi, 1], "-", lw=1.4)
                    ax.plot(pos[i, p, :Oi, 0], pos[i, p, :Oi, 1], "--", lw=1.1)
            ax.set_aspect("equal");
            ax.set_title(f"play {i}")
        fig.suptitle("true (solid) vs predicted (dashed)")
        fig.tight_layout();
        fig.savefig(fig_dir / "trajectories.png");
        plt.close(fig)

    traj_examples(model, te_loader, device)
    print("baseline figures saved to", fig_dir)


def plot_tuning(history_baseline, tune_results, n_base, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .3})

    # (1) Val curves
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history_baseline["val_rmse"], lw=2, color="#888888",
            label=f"baseline (best {min(history_baseline['val_rmse']):.3f})")
    for r in tune_results:
        ax.plot(r["history"]["val_rmse"], label=f"{r['name']} (best {r['best_val_rmse']:.3f})")
    ax.set_xlabel("epoch");
    ax.set_ylabel("val RMSE (yards)")
    ax.set_title("Hyperparameter tuning (Randomized Search)");
    ax.legend(fontsize=8)
    fig.tight_layout();
    fig.savefig(fig_dir / "tuning_val_curves.png");
    plt.close(fig)

    # (2) Best Val RMSE Comparison
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = ["baseline"] + [r["name"] for r in tune_results]
    vals = [min(history_baseline["val_rmse"])] + [r["best_val_rmse"] for r in tune_results]
    bars = ax.bar(names, vals, color=["#888888", "#4c72b0", "#55a868", "#c44e52", "#9467bd", "#d62728"][:len(names)])
    ax.axhline(min(history_baseline["val_rmse"]), ls="--", lw=1, color="#888888")
    for bb, v in zip(bars, vals):
        ax.text(bb.get_x() + bb.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("best val RMSE (yards)")
    ax.set_title("Best Val RMSE per Configuration")
    fig.tight_layout();
    fig.savefig(fig_dir / "tuning_comparison.png");
    plt.close(fig)

    # (3) Params vs Quality
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter([n_base / 1e6], [min(history_baseline["val_rmse"])], s=60, color="#888888")
    ax.annotate("baseline", (n_base / 1e6, min(history_baseline["val_rmse"])), xytext=(6, 4),
                textcoords="offset points", fontsize=8)
    for r in tune_results:
        ax.scatter([r["n_params"] / 1e6], [r["best_val_rmse"]], s=60)
        ax.annotate(r["name"], (r["n_params"] / 1e6, r["best_val_rmse"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("params (M)");
    ax.set_ylabel("best val RMSE (yards)")
    ax.set_title("Capacity vs Quality");
    fig.tight_layout()
    fig.savefig(fig_dir / "tuning_params_vs_val.png");
    plt.close(fig)
    print("tuning figures saved to", fig_dir)