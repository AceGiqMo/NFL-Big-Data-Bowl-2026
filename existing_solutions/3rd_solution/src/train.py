from __future__ import annotations

import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).parent))
from model import STTransformer
from losses import total_loss


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PlayDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.items = pickle.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def split_by_game(items, val_frac=0.2, seed=42):
    games = sorted({x["game_id"] for x in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    n_val_games = max(1, int(round(len(games) * val_frac)))
    val_games = set(games[:n_val_games])
    train = [x for x in items if x["game_id"] not in val_games]
    val = [x for x in items if x["game_id"] in val_games]
    if not train:
        raise ValueError("Game-level split produced an empty training set.")
    return train, val


def collate(batch):
    H = max(x["horizon"] for x in batch)
    X = torch.tensor(np.stack([x["X"] for x in batch]), dtype=torch.float32)
    Y = torch.zeros(len(batch), H, 22, 2)
    M = torch.zeros(len(batch), H, 22)
    for i, item in enumerate(batch):
        h = item["horizon"]
        Y[i, :h] = torch.tensor(item["Y"], dtype=torch.float32)
        M[i, :h] = torch.tensor(item["target_frame_mask"], dtype=torch.float32)
    PM = torch.tensor(np.stack([x["player_mask"] for x in batch]), dtype=torch.float32)
    TM = torch.tensor(np.stack([x["time_mask"] for x in batch]), dtype=torch.float32)
    TG = torch.tensor(np.stack([x["target_mask"] for x in batch]), dtype=torch.float32)
    return X, Y, M, PM, TM, TG


def rmse_sums(pred, target, mask):
    """Return global squared-error sum and coordinate count for RMSE."""
    m = mask.unsqueeze(-1)
    squared_error = ((pred - target) ** 2 * m).sum()
    coord_count = m.sum() * 2.0
    return squared_error, coord_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default="checkpoints/model.pt")
    args = ap.parse_args()

    seed_everything(args.seed)
    with open(args.data, "rb") as f:
        items = pickle.load(f)
    if len(items) < 5:
        raise ValueError("Need at least 5 plays for a meaningful train/validation split.")

    train_items, val_items = split_by_game(items, args.val_frac, args.seed)
    train_ds, val_ds = PlayDataset.__new__(PlayDataset), PlayDataset.__new__(PlayDataset)
    train_ds.items, val_ds.items = train_items, val_items

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heads = args.heads
    while args.hidden % heads != 0 and heads > 1:
        heads -= 1
    max_h = max(x["horizon"] for x in items)
    model = STTransformer(
        train_items[0]["X"].shape[-1],
        hidden=args.hidden,
        heads=heads,
        layers=args.layers,
        max_time=train_items[0]["X"].shape[0],
        horizon=max_h,
    ).to(device)

    # Keep the simple optimizer from the write-up; use cosine annealing.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=1e-5
    )

    best = float("inf")
    history = []
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for X, Y, M, PM, TM, TG in dl:
            X, Y, M, PM, TM, TG = [z.to(device) for z in (X, Y, M, PM, TM, TG)]
            opt.zero_grad(set_to_none=True)
            outputs = model(X, TM, PM)
            # Different plays can have different horizons. The model emits the
            # dataset-wide maximum horizon; only the valid batch horizon belongs
            # in this loss.
            outputs["main"] = outputs["main"][:, :Y.size(1)]
            loss, _ = total_loss(outputs, Y, M, X, TM, PM, TG)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * X.size(0)

        model.eval()
        val_sse = torch.tensor(0.0, device=device)
        val_count = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for X, Y, M, PM, TM, TG in vdl:
                X, Y, M, PM, TM, TG = [z.to(device) for z in (X, Y, M, PM, TM, TG)]
                outputs = model(X, TM, PM)
                pred = outputs["main"][:, :Y.size(1)]
                batch_sse, batch_count = rmse_sums(pred, Y, M)
                val_sse += batch_sse
                val_count += batch_count
        val_rmse = float(torch.sqrt(val_sse / val_count.clamp_min(1.0)))
        sch.step()
        train_loss = running / max(len(train_ds), 1)
        lr_now = sch.get_last_lr()[0]
        history.append({"epoch": ep, "train_loss": train_loss, "val_rmse": val_rmse, "lr": lr_now})
        print(
            f"epoch {ep:02d} train_loss={train_loss:.5f} "
            f"val_RMSE={val_rmse:.5f} lr={lr_now:.2e}"
        )

        if val_rmse < best:
            best = val_rmse
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_dim": train_items[0]["X"].shape[-1],
                    "hidden": args.hidden,
                    "heads": heads,
                    "layers": args.layers,
                    "max_time": train_items[0]["X"].shape[0],
                    "horizon": max_h,
                    "val_rmse": best,
                },
                out_path,
            )

    history_path = Path("outputs/training_history.csv")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_rmse", "lr"])
        writer.writeheader()
        writer.writerows(history)
    print(f"saved best checkpoint: {out_path} (val_RMSE={best:.5f})")
    print(f"saved training history: {history_path}")


if __name__ == "__main__":
    main()
