from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from train import split_by_game


def evaluate(items, mode: str) -> float:
    sse = 0.0
    count = 0.0
    for item in items:
        h = item["horizon"]
        mask = item["target_frame_mask"].astype(bool)
        if mode == "stationary":
            pred = np.zeros_like(item["Y"], dtype=np.float32)
        elif mode == "constant_velocity":
            # 10 Hz tracking: one frame is 0.1 seconds.
            t = np.arange(1, h + 1, dtype=np.float32)[:, None, None]
            vx = item["X"][-1, :, 6]
            vy = item["X"][-1, :, 7]
            pred = np.stack([vx[None, :] * 0.1 * t[:, :, 0],
                             vy[None, :] * 0.1 * t[:, :, 0]], axis=-1)
        else:
            raise ValueError(f"Unknown baseline: {mode}")

        err2 = (pred - item["Y"]) ** 2
        sse += float(err2[mask].sum())
        count += float(mask.sum() * 2)

    if count == 0:
        raise ValueError("No valid target coordinates found.")
    return float(np.sqrt(sse / count))


def main():
    ap = argparse.ArgumentParser(description="Evaluate simple trajectory baselines.")
    ap.add_argument("--data", required=True, help="Processed pickle from preprocess.py")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.data, "rb") as f:
        items = pickle.load(f)
    _, val_items = split_by_game(items, args.val_frac, args.seed)

    print(f"validation plays: {len(val_items)}")
    print(f"stationary RMSE:       {evaluate(val_items, 'stationary'):.5f}")
    print(f"constant velocity RMSE:{evaluate(val_items, 'constant_velocity'):.5f}")


if __name__ == "__main__":
    main()
