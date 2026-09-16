from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="outputs/training_history.csv")
    ap.add_argument("--output", default="outputs/training_curves.png")
    args = ap.parse_args()

    df = pd.read_csv(args.history)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(df["epoch"], df["val_rmse"], marker="o")
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (yards)")
    axes[1].grid(True, alpha=0.25)

    fig.suptitle("NFL trajectory forecasting — training curves")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    print(args.output)


if __name__ == "__main__":
    main()
