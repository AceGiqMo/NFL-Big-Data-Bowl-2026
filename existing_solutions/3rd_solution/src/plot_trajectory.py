"""Plot one player's observed history, ground-truth future, and predicted future.

Usage:
    python plot_trajectory.py --data data/processed/dev.pkl \
        --checkpoint checkpoints/dev.pt --index 0 --player-slot 0
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from model import STTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--index", type=int, default=0, help="which play in the pickle to plot")
    p.add_argument("--player-slot", type=int, default=0, help="which of the 22 fixed slots to plot")
    p.add_argument("--output", default="outputs/trajectory.png")
    return p.parse_args()


def load_model(checkpoint_path):
    ck = torch.load(checkpoint_path, map_location="cpu")
    model = STTransformer(
        ck["in_dim"],
        ck["hidden"],
        heads=ck.get("heads", max(1, ck["hidden"] // 32)),
        layers=ck["layers"],
        max_time=ck["max_time"],
        horizon=ck["horizon"],
    )
    model.load_state_dict(ck["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    with open(args.data, "rb") as f:
        items = pickle.load(f)
    sample = items[args.index]
    model = load_model(args.checkpoint)

    X = torch.tensor(sample["X"][None], dtype=torch.float32)
    player_mask = torch.tensor(sample["player_mask"][None])
    time_mask = torch.tensor(sample["time_mask"][None])

    with torch.no_grad():
        pred_disp = model(X, time_mask, player_mask)["main"][0, :, args.player_slot].numpy()
    pred_xy = pred_disp + sample["last_xy"][args.player_slot]

    target_xy = sample["Y"][:, args.player_slot] + sample["last_xy"][args.player_slot]
    target_valid = sample["target_frame_mask"][:, args.player_slot] > 0

    # IMPORTANT: short plays are left-padded with zeros up to the fixed window
    # (see preprocess.py). Those padded frames must be dropped here, otherwise
    # the plot draws a fake trajectory segment starting at (0, 0).
    time_valid = sample["time_mask"] > 0
    hist_xy = sample["X"][time_valid, args.player_slot, :2]

    plt.figure(figsize=(7, 6))
    plt.plot(hist_xy[:, 0], hist_xy[:, 1], marker="o", label="history (observed)")
    if target_valid.any():
        plt.plot(target_xy[target_valid, 0], target_xy[target_valid, 1], marker="o", label="actual future")
        n_valid = int(target_valid.sum())
        plt.plot(pred_xy[:n_valid, 0], pred_xy[:n_valid, 1], marker="o", label="predicted future")
    else:
        plt.plot(pred_xy[:, 0], pred_xy[:, 1], marker="o", label="predicted future")

    plt.xlabel("x (yards)")
    plt.ylabel("y (yards)")
    plt.title(f'Play {sample["game_id"]}-{sample["play_id"]}, player slot {args.player_slot}')
    plt.legend()
    plt.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(args.output)


if __name__ == "__main__":
    main()
