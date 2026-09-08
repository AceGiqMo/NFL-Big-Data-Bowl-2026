from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from features import add_features, FEATURES


def load_week(data_dir: Path, week: int):
    inp = pd.read_csv(data_dir / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(data_dir / f"output_2023_w{week:02d}.csv")
    return add_features(inp), out


def process_play(inp: pd.DataFrame, out: pd.DataFrame, window: int):
    inp = inp.sort_values(["frame_id", "nfl_id"]).reset_index(drop=True)
    out = out.sort_values(["frame_id", "nfl_id"]).reset_index(drop=True)

    game_id = int(inp.game_id.iloc[0])
    play_id = int(inp.play_id.iloc[0])
    all_frames = sorted(inp.frame_id.unique().tolist())
    if not all_frames:
        return None

    # Use the last `window` observed frames, left-padding if a play is shorter.
    keep = all_frames[-window:]
    T = window
    time_offset = T - len(keep)
    first = inp[inp.frame_id == all_frames[0]].copy()

    passer = first[first.is_passer > 0.5][["x", "y"]]
    if len(passer):
        qx, qy = float(passer.iloc[0].x), float(passer.iloc[0].y)
    else:
        qx, qy = float(first.x.mean()), float(first.y.mean())

    # Public write-up: offense slots 0..10, defense slots 11..21,
    # sorted once by distance to passer at frame_id=1.
    first["qb_dist"] = np.hypot(first.x - qx, first.y - qy)
    first["side_order"] = (first.player_side.astype(str).str.lower() != "offense").astype(int)
    first = first.sort_values(["side_order", "qb_dist", "nfl_id"])
    ids = first.nfl_id.astype(int).tolist()
    slot_map = {nfl_id: slot for slot, nfl_id in enumerate(ids[:22])}
    if not slot_map:
        return None

    F = len(FEATURES)
    X = np.zeros((T, 22, F), dtype=np.float32)
    time_mask = np.zeros(T, dtype=np.float32)
    player_mask = np.zeros(22, dtype=np.float32)
    target_mask = np.zeros(22, dtype=np.float32)
    last_xy = np.zeros((22, 2), dtype=np.float32)

    # Build the observed tensor.
    for ti, frame in enumerate(keep):
        dst_t = time_offset + ti
        g = inp[inp.frame_id == frame]
        time_mask[dst_t] = 1.0
        for r in g.itertuples(index=False):
            sid = slot_map.get(int(r.nfl_id))
            if sid is None:
                continue
            vals = np.asarray([getattr(r, c) for c in FEATURES], dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            X[dst_t, sid] = vals
            player_mask[sid] = 1.0
            if frame == keep[-1]:
                last_xy[sid] = [float(r.x), float(r.y)]
                target_mask[sid] = float(bool(r.player_to_predict))

    # Use the target flag from input rather than inferring it from output rows.
    if target_mask.sum() == 0:
        return None

    horizon = int(out.loc[out.nfl_id.isin(slot_map.keys()), "frame_id"].max()) if len(out) else 0
    if horizon <= 0:
        horizon = int(inp["num_frames_output"].max())
    if horizon <= 0:
        return None

    Y_abs = np.zeros((horizon, 22, 2), dtype=np.float32)
    target_frame_mask = np.zeros((horizon, 22), dtype=np.float32)
    for r in out.itertuples(index=False):
        sid = slot_map.get(int(r.nfl_id))
        if sid is None or target_mask[sid] < 0.5:
            continue
        k = int(r.frame_id) - 1
        if 0 <= k < horizon:
            Y_abs[k, sid] = [float(r.x), float(r.y)]
            target_frame_mask[k, sid] = 1.0

    # The model target is future displacement from the final observed point.
    Y_disp = Y_abs - last_xy[None, :, :]

    return {
        "game_id": game_id,
        "play_id": play_id,
        "X": X,
        "Y": Y_disp,
        "Y_abs": Y_abs,
        "player_mask": player_mask,
        "target_mask": target_mask,
        "target_frame_mask": target_frame_mask,
        "time_mask": time_mask,
        "last_xy": last_xy,
        "horizon": horizon,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--weeks", nargs="+", type=int, required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--max-plays", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    samples = []
    for week in args.weeks:
        inp, out = load_week(data_dir, week)
        for (gid, pid), g in inp.groupby(["game_id", "play_id"], sort=False):
            og = out[(out.game_id == gid) & (out.play_id == pid)]
            sample = process_play(g, og, args.window)
            if sample is not None:
                samples.append(sample)
            if args.max_plays and len(samples) >= args.max_plays:
                break
        if args.max_plays and len(samples) >= args.max_plays:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(samples)} plays -> {args.output}")


if __name__ == "__main__":
    main()
