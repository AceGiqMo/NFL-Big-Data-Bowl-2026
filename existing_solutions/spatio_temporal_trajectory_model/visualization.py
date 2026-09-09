"""
visualization.py — pure plotting utilities (no training logic, no model imports).

Every function takes plain numpy/dict data plus an output path, saves a PNG and
closes the figure, so train.py can call them in any order. Matplotlib runs in
the headless "Agg" backend (server-friendly, reproducible renders).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

ROLE_COLORS = {"Targeted Receiver": "#c44e52",
               "Defensive Coverage": "#4c72b0",
               "Passer": "#55a868",
               "Other Route Runner": "#999999"}


def plot_training_history(history: dict, path: Path) -> None:
    """Loss curves + validation RMSE with the best-epoch marker."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train huber")
    axes[0].plot(history["val_loss"], label="val huber (EMA)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss curves")
    axes[0].legend()
    axes[1].plot(history["val_rmse"], color="#c44e52", label="val RMSE (yards)")
    if history.get("best_epoch", -1) >= 0:
        axes[1].axvline(history["best_epoch"], ls=":", c="gray", label="best epoch")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("Validation RMSE (EMA weights)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rmse_comparison(results: Dict[str, dict], path: Path) -> None:
    """Bar chart of test RMSE with 95% bootstrap confidence intervals."""
    names = list(results)
    point = np.array([results[n]["rmse"] for n in names])
    lo = np.array([results[n]["ci_lo"] for n in names])
    hi = np.array([results[n]["ci_hi"] for n in names])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(len(names))
    ax.bar(xs, point, yerr=np.vstack([point - lo, hi - point]), capsize=6,
           color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][:len(names)])
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("test RMSE (yards)")
    ax.set_title("Test-week comparison (95% CI, block bootstrap)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_rmse_vs_horizon(per_frame: Dict[str, Tuple[np.ndarray, np.ndarray]],
                         path: Path, bin_size: int = 4) -> None:
    """RMSE(o): error growth over the prediction horizon (cumsum drift)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, (h, e) in per_frame.items():
        if h.size == 0:
            continue
        bins = np.arange(0, h.max() + bin_size, bin_size)
        idx = np.digitize(h, bins) - 1
        centers, means = [], []
        for i in range(len(bins) - 1):
            sel = idx == i
            if sel.any():
                centers.append(bins[i] + bin_size / 2)
                means.append(np.sqrt(e[sel].mean() / 2.0))
        ax.plot(centers, means, marker="o", ms=4, label=name)
    ax.set_xlabel("future frame index o (0.1 s per frame)")
    ax.set_ylabel("RMSE(o) (yards)")
    ax.set_title("Error growth over prediction horizon")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_role_breakdown(role_stats: Dict[str, Dict[str, np.ndarray]],
                        role_order: List[str], path: Path) -> None:
    """Per-player RMSE distributions by role (box plots, outliers hidden)."""
    data, labels, colors = [], [], []
    for name, per_role in role_stats.items():
        for role in role_order:
            if role in per_role and per_role[role].size:
                data.append(per_role[role])
                labels.append(f"{name}\n{role}")
                colors.append(ROLE_COLORS.get(role, "#999999"))
    if not data:
        return
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(data)), 4.2))
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("per-player RMSE (yards)")
    ax.set_title("Error distribution by player role (test weeks)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_trajectories(samples: List[dict], path: Path) -> None:
    """True (solid) vs predicted (dashed) trajectories; star = ball landing."""
    n = min(4, len(samples))
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.3), squeeze=False)
    for ax, s in zip(axes[0], samples[:n]):
        for role, true, pred, last in s["players"]:
            c = ROLE_COLORS.get(role, "#999999")
            ax.plot(true[:, 0], true[:, 1], "-", c=c, lw=1.4)
            ax.plot(pred[:, 0], pred[:, 1], "--", c=c, lw=1.1)
            ax.plot(last[0], last[1], "o", c=c, ms=4)
        ax.plot(s["land"][0], s["land"][1], "*", c="gold", ms=14, mec="k")
        ax.set_aspect("equal")
        ax.set_title(f"play {s['play']}")
        ax.set_xlabel("x (yards)")
        ax.set_ylabel("y (yards)")
    fig.suptitle("True (solid) vs predicted (dashed) trajectories")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_error_qq(radial_errors: np.ndarray, path: Path) -> None:
    """Percentile curve of radial errors: heavy tails justify the Huber loss."""
    qs = np.linspace(1, 99, 99)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(qs, np.percentile(radial_errors, qs), marker=".", ms=3, ls="none")
    ax.set_xlabel("percentile")
    ax.set_ylabel("radial error (yards)")
    ax.set_title("Error distribution (heavy tails -> robust loss)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_eda(df, path: Path) -> None:
    """Dataset statistics: input/output length and player-count histograms."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, col in zip(axes, ["S", "O", "P", "scored"]):
        ax.hist(df[col], bins=40, color="#4c72b0")
        ax.set_title(f"{col} per play")
        ax.set_xlabel(col)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_seed_summary(per_exp_rmse: Dict[str, float], path: Path) -> None:
    """Per-seed test RMSE points with mean +/- std band (multi-seed runs)."""
    names = list(per_exp_rmse)
    vals = np.array([per_exp_rmse[n] for n in names])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(np.arange(len(names)), vals, color="#4c72b0", alpha=0.8)
    if len(vals) > 1:
        ax.axhline(vals.mean(), c="#c44e52", ls="--",
                   label=f"mean {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
        ax.fill_between([-0.5, len(names) - 0.5],
                        vals.mean() - vals.std(ddof=1),
                        vals.mean() + vals.std(ddof=1),
                        color="#c44e52", alpha=0.12)
        ax.legend()
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=12)
    ax.set_ylabel("test RMSE (yards)")
    ax.set_title("Seed-to-seed stability of the proposed model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)