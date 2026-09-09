"""Small, model-oriented EDA for NFL Big Data Bowl 2026.

Usage:
    python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda

The script reads one week only, because the full competition data are ~865 MB.
It produces a compact set of tables and plots that are useful for explaining
why trajectory forecasting needs temporal + inter-player modeling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_hist(df, col, out):
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=50)
    ax.set_title(col)
    ax.set_xlabel(col)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / f"{col}.png", dpi=140)
    plt.close(fig)


def trajectory_plot(inp, out_df, out):
    key = inp[["game_id", "play_id"]].drop_duplicates().iloc[0]
    gid, pid = int(key.game_id), int(key.play_id)
    x = inp[(inp.game_id == gid) & (inp.play_id == pid)].copy()
    y = out_df[(out_df.game_id == gid) & (out_df.play_id == pid)].copy()

    first_frame = x.frame_id.min()
    players = x[x.frame_id == first_frame]["nfl_id"].head(6).tolist()
    fig, ax = plt.subplots(figsize=(9, 6))
    for nfl_id in players:
        hist = x[x.nfl_id == nfl_id].sort_values("frame_id")
        ax.plot(hist.x, hist.y, alpha=0.7)
        fut = y[y.nfl_id == nfl_id].sort_values("frame_id")
        if len(fut):
            ax.plot(fut.x, fut.y, linestyle="--", alpha=0.8)
    ax.set_title(f"Example play trajectories: game={gid}, play={pid}")
    ax.set_xlabel("x (yards)")
    ax.set_ylabel("y (yards)")
    fig.tight_layout()
    fig.savefig(out / "example_trajectories.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--output-dir", default="outputs/eda")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    inp_path = data_dir / f"input_2023_w{args.week:02d}.csv"
    out_path = data_dir / f"output_2023_w{args.week:02d}.csv"
    if not inp_path.exists() or not out_path.exists():
        raise FileNotFoundError(f"Missing {inp_path} or {out_path}")

    inp = pd.read_csv(inp_path)
    fut = pd.read_csv(out_path)

    print("=== SHAPES ===")
    print("input:", inp.shape)
    print("output:", fut.shape)

    print("\n=== INPUT COLUMNS ===")
    print(inp.columns.tolist())

    print("\n=== MISSING VALUES (%) ===")
    print((inp.isna().mean() * 100).sort_values(ascending=False).head(15).round(3))

    plays = inp[["game_id", "play_id"]].drop_duplicates()
    players_per_play = inp.groupby(["game_id", "play_id"])["nfl_id"].nunique()
    frames_per_play = inp.groupby(["game_id", "play_id"])["frame_id"].nunique()
    horizon = inp.groupby(["game_id", "play_id"])["num_frames_output"].first()
    target_players = inp[inp["player_to_predict"].astype(bool)].groupby(["game_id", "play_id"])["nfl_id"].nunique()

    print("\n=== PLAY-LEVEL STATS ===")
    print(f"unique plays: {len(plays):,}")
    print("players/play:")
    print(players_per_play.describe().round(2))
    print("observed frames/play:")
    print(frames_per_play.describe().round(2))
    print("future frames/play:")
    print(horizon.describe().round(2))
    print("target players/play:")
    print(target_players.describe().round(2))

    print("\n=== ROLES ===")
    print(inp["player_role"].value_counts(dropna=False))

    print("\n=== POSITIONS ===")
    print(inp["player_position"].value_counts(dropna=False).head(15))

    print("\n=== SIMPLE NUMERIC SUMMARY ===")
    cols = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "num_frames_output"]
    print(inp[cols].describe().T.round(3))

    for col in ["x", "y", "s", "a", "dir", "o"]:
        save_hist(inp, col, out)
    trajectory_plot(inp, fut, out)

    # A useful relationship for the report: distance from receiver to landing point.
    rec = inp[inp["player_role"].eq("Targeted Receiver")][
        ["game_id", "play_id", "frame_id", "x", "y", "ball_land_x", "ball_land_y"]
    ].copy()
    rec["distance_to_ball_land"] = np.hypot(
        rec.x - rec.ball_land_x, rec.y - rec.ball_land_y
    )
    final_rec = rec.sort_values("frame_id").groupby(["game_id", "play_id"]).tail(1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(final_rec["distance_to_ball_land"].dropna(), bins=40)
    ax.set_title("Targeted receiver distance to ball landing at last input frame")
    ax.set_xlabel("distance (yards)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out / "receiver_distance_to_ball_land.png", dpi=140)
    plt.close(fig)

    summary = {
        "week": args.week,
        "input_rows": len(inp),
        "output_rows": len(fut),
        "plays": len(plays),
        "mean_players_per_play": float(players_per_play.mean()),
        "mean_observed_frames": float(frames_per_play.mean()),
        "mean_future_frames": float(horizon.mean()),
        "mean_target_players": float(target_players.mean()),
    }
    pd.Series(summary).to_csv(out / "summary.csv", header=False)
    print(f"\nSaved EDA outputs to: {out}")


if __name__ == "__main__":
    main()
