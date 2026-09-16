from __future__ import annotations

import numpy as np
import pandas as pd


def _num(d: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(d[col], errors="coerce").astype("float32")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    numeric = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "num_frames_output", "frame_id"]
    for c in numeric:
        d[c] = _num(d, c).fillna(0.0)

    # NFL tracking angles are in degrees. We keep the original angle feature,
    # and additionally expose a radian representation for trig operations.
    d["dir_rad"] = np.deg2rad(d["dir"]).astype("float32")
    d["velocity_x"] = (d["s"] * np.cos(d["dir_rad"])).astype("float32")
    d["velocity_y"] = (d["s"] * np.sin(d["dir_rad"])).astype("float32")
    d["angle_to_ball"] = np.arctan2(
        d["ball_land_y"] - d["y"], d["ball_land_x"] - d["x"]
    ).astype("float32")

    role = d["player_role"].astype(str).str.strip().str.lower()
    side = d["player_side"].astype(str).str.strip().str.lower()
    d["player_side_bool"] = (side == "offense").astype("float32")
    d["is_passer"] = (role == "passer").astype("float32")
    d["is_receiver"] = (role == "targeted receiver").astype("float32")

    # Passer/receiver coordinates at the same frame.
    keys = ["game_id", "play_id", "frame_id"]
    passer = (
        d.loc[d["is_passer"] > 0.5]
        .groupby(keys, sort=False)[["x", "y"]]
        .first()
        .rename(columns={"x": "passer_x", "y": "passer_y"})
    )
    receiver = (
        d.loc[d["is_receiver"] > 0.5]
        .groupby(keys, sort=False)[["x", "y"]]
        .first()
        .rename(columns={"x": "receiver_x", "y": "receiver_y"})
    )
    d = d.merge(passer, on=keys, how="left").merge(receiver, on=keys, how="left")
    for c, fallback in [
        ("passer_x", "x"), ("passer_y", "y"),
        ("receiver_x", "x"), ("receiver_y", "y"),
    ]:
        d[c] = d[c].fillna(d[fallback])

    d["distance_to_passer"] = np.hypot(d["x"] - d["passer_x"], d["y"] - d["passer_y"])
    d["distance_to_receiver"] = np.hypot(d["x"] - d["receiver_x"], d["y"] - d["receiver_y"])
    d["distance_to_ball_land"] = np.hypot(d["x"] - d["ball_land_x"], d["y"] - d["ball_land_y"])
    d["passer_to_ball_land"] = np.hypot(d["passer_x"] - d["ball_land_x"], d["passer_y"] - d["ball_land_y"])
    d["receiver_to_ball_land"] = np.hypot(d["receiver_x"] - d["ball_land_x"], d["receiver_y"] - d["ball_land_y"])

    # Geometry relative to the passer-receiver segment.
    ax, ay = d["passer_x"].to_numpy(), d["passer_y"].to_numpy()
    bx, by = d["receiver_x"].to_numpy(), d["receiver_y"].to_numpy()
    px, py = d["x"].to_numpy(), d["y"].to_numpy()
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby + 1e-6
    proj = ((px - ax) * abx + (py - ay) * aby) / denom
    proj_clip = np.clip(proj, 0.0, 1.0)
    proj_x, proj_y = ax + proj_clip * abx, ay + proj_clip * aby
    d["distance_to_passing_line"] = np.hypot(px - proj_x, py - proj_y).astype("float32")
    d["projection_on_passing_line"] = proj.astype("float32")
    d["triangle_area_ratio"] = (
        np.abs(abx * (py - ay) - aby * (px - ax)) / (np.hypot(abx, aby) + 1e-6)
    ).astype("float32")

    # Relative time inside the observed play is more meaningful than a global frame id.
    d["time_elapsed"] = (d["frame_id"] - 1.0) * 0.1
    return d


FEATURES = [
    "x", "y", "s", "a", "dir_rad", "o",
    "velocity_x", "velocity_y", "angle_to_ball",
    "ball_land_x", "ball_land_y", "num_frames_output",
    "player_side_bool", "is_passer", "is_receiver",
    "distance_to_passer", "distance_to_receiver", "distance_to_ball_land",
    "passer_to_ball_land", "receiver_to_ball_land",
    "distance_to_passing_line", "projection_on_passing_line", "triangle_area_ratio",
    "time_elapsed",
]
