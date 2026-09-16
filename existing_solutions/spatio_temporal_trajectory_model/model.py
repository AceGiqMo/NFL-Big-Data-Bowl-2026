"""
model.py — configuration, data pipeline and model definition.

Contents
    * paths / config / seeding               (reproducibility)
    * NFL tracking data loading & padding    (official week-based splits)
    * architecture: delta-prediction spatio-temporal model
      (SqueezeFormer temporal blocks, distance-biased spatial attention,
      RoPE over output horizons, BiLSTM || BiGRU, refinement blocks)
    * Muon / AdamW optimizers, EMA, Huber delta loss, official RMSE

Bug-fix history (ported from the Kaggle training session)
    FIX-1  deltas_to_absolute: anchor broadcast (..., 2) -> (..., 1, 2).
    FIX-2  output-phase features: every horizon feature expanded over players.
    FIX-3  NaN-safe attention: softmax over fully-masked rows (padded player /
           padded frame) produced NaN; padded tokens are now REPLACED with
           zeros via torch.where (multiplication cannot remove NaN).
    FIX-4  optimizer stability: Muon is opt-in (use_muon=False by default),
           optimizers are returned as a list so train() handles 1 or 2 of them.
    FIX-5  AMP is opt-in (amp=False by default); loss computed in float32.

This module has NO entry point; it is imported by train.py.
References (mandatory attribution in the report):
    [1] NFL Big Data Bowl 2026 - Prediction (Kaggle, data & task).
    [2] Public 5th-place solution summary (architecture ideas).
    [3] Muon optimizer: https://github.com/KellerJordan/Muon
    [4] RoPE: Su et al., 2021.   [5] SqueezeFormer: Kim et al., 2022.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# Paths (derived from this file's location -> structure-agnostic)
# --------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJ_ROOT / "data"
OUT_ROOT = PROJ_ROOT / "outputs"
FIG_ROOT = PROJ_ROOT / "figures"
CKPT_ROOT = PROJ_ROOT / "checkpoints"

ROLE_LIST = ["Defensive Coverage", "Targeted Receiver", "Passer", "Other Route Runner"]


def exp_dirs(exp_name: str) -> Tuple[Path, Path]:
    """Per-experiment output/figure directories (created on demand)."""
    out, fig = OUT_ROOT / exp_name, FIG_ROOT / exp_name
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    return out, fig


def ckpt_path(exp_name: str) -> Path:
    """Path of the best-EMA checkpoint for a given experiment name."""
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    return CKPT_ROOT / f"{exp_name}_best.pt"


# --------------------------------------------------------------------------- #
# Config & seeding
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    # --- data / split (official-style: train on early weeks, val/test later)
    data_dir: str = str(DATA_DIR)
    train_weeks: List[int] = field(default_factory=lambda: list(range(1, 16)))
    val_weeks: List[int] = field(default_factory=lambda: [16, 17])
    test_weeks: List[int] = field(default_factory=lambda: [18])
    max_input_frames: int = 64       # S: cap on pre-pass history length
    max_output_frames: int = 48      # O: cap on predicted horizon
    min_output_frames: int = 5       # competition drops <0.5s passes anyway
    # --- model dimensions
    d_model: int = 128
    n_heads: int = 8
    n_temporal_blocks: int = 4       # EncoderBlockV1 count
    n_spatial_layers_pre: int = 2    # spatial Transformer at the release frame
    n_refine_blocks: int = 2         # EncoderBlockV2 count
    rnn_hidden: int = 96             # per-direction hidden of BiLSTM/BiGRU
    dropout: float = 0.1
    dist_bias_buckets: int = 16      # learned distance-matrix attention bias
    dist_bias_max: float = 40.0      # yards, beyond -> last bucket
    rope_base: float = 10000.0
    # --- training
    seed: int = 42
    epochs: int = 40
    batch_size: int = 32
    lr_muon: float = 0.02
    lr_adamw: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    huber_delta: float = 0.35        # yards/frame: quadratic/linear switch
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    patience: int = 6                # early stopping on val RMSE
    num_workers: int = 4
    amp: bool = False                # FIX-5: fp16 autocast is opt-in
    use_muon: bool = False           # FIX-4: Muon is opt-in (AdamW everywhere)
    # --- evaluation / statistics
    n_bootstrap: int = 1000
    bootstrap_block: int = 8         # block size in plays (temporal dependence)
    # --- misc
    device: str = "auto"
    exp_name: str = "main"

    def resolve_device(self) -> torch.device:
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Fix all RNGs and make cuDNN deterministic (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def _norm_angle(deg: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0


def load_week(data_dir: Path, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load input/output CSVs for one week."""
    inp = pd.read_csv(data_dir / f"input_2023_w{week:02d}.csv")
    out = pd.read_csv(data_dir / f"output_2023_w{week:02d}.csv")
    return inp, out


def build_plays(inp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, play) holding the raw frame tables."""
    inp = inp.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    out = out.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    g_out = out.groupby(["game_id", "play_id"], sort=False)
    rows = []
    for (gid, pid), gi in inp.groupby(["game_id", "play_id"], sort=False):
        try:
            go = g_out.get_group((gid, pid))
        except KeyError:
            continue  # play without output frames -> unusable
        first = gi.iloc[0]
        rows.append(dict(game_id=gid, play_id=pid, inp=gi, out=go,
                         ball_land_x=float(first.ball_land_x),
                         ball_land_y=float(first.ball_land_y)))
    return pd.DataFrame(rows)


def role_codes(inp: pd.DataFrame, players: np.ndarray) -> np.ndarray:
    """Integer role code per player index (for per-role statistics)."""
    mapping = inp.drop_duplicates("nfl_id").set_index("nfl_id").player_role
    return np.array([ROLE_LIST.index(mapping[p]) if mapping[p] in ROLE_LIST else 3
                     for p in players], np.int64)


def prepare_play(inp: pd.DataFrame, out: pd.DataFrame,
                 cfg: Config) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert one play to padded-ready float arrays.

    Convention: if the offense moves left, x/y and horizontal angles are
    flipped so the offense always attacks +x. Targets are per-frame deltas
    in this normalized frame, anchored at the release-frame position.
    """
    players = inp.nfl_id.unique()
    P = len(players)
    pidx = {p: i for i, p in enumerate(players)}
    flip = inp.play_direction.iloc[0] == "left"
    ang_shift = 180.0 if flip else 0.0

    # ---- input phase ------------------------------------------------------
    S = min(int(inp.frame_id.max()), cfg.max_input_frames)
    inp = inp[inp.frame_id <= S]
    X = np.zeros((P, S, 2), np.float32)
    SA = np.zeros((P, S), np.float32)
    AA = np.zeros((P, S), np.float32)
    OO = np.zeros((P, S), np.float32)
    DD = np.zeros((P, S), np.float32)
    to_pred = np.zeros(P, np.float32)
    side = np.zeros(P, np.float32)
    is_target = np.zeros(P, np.float32)
    is_passer = np.zeros(P, np.float32)

    for nfl_id, gi in inp.groupby("nfl_id", sort=False):
        i = pidx[nfl_id]
        f = np.clip(gi.frame_id.to_numpy() - 1, 0, S - 1)
        x = gi.x.to_numpy(np.float32)
        y = gi.y.to_numpy(np.float32)
        if flip:
            x, y = 120.0 - x, 53.3 - y
        X[i, f, 0], X[i, f, 1] = x, y
        SA[i, f] = gi.s.to_numpy(np.float32)
        AA[i, f] = gi.a.to_numpy(np.float32)
        OO[i, f] = _norm_angle(gi.o.to_numpy(np.float32) + ang_shift)
        DD[i, f] = _norm_angle(gi.dir.to_numpy(np.float32) + ang_shift)
        to_pred[i] = float(gi.player_to_predict.iloc[0])
        side[i] = float(gi.player_side.iloc[0] == "Offense")
        is_target[i] = float(gi.player_role.iloc[0] == "Targeted Receiver")
        is_passer[i] = float(gi.player_role.iloc[0] == "Passer")

    dX = np.zeros_like(X)
    dX[:, 1:] = X[:, 1:] - X[:, :-1]                       # input-phase deltas
    t = np.arange(S, dtype=np.float32)
    time_decay = np.exp(-(S - 1 - t) / 10.0)[:, None]      # (S, 1) recency weight

    # ---- static geometry at the release frame -----------------------------
    last_xy = X[:, -1].copy()
    land = np.array([inp.ball_land_x.iloc[0], inp.ball_land_y.iloc[0]], np.float32)
    if flip:
        land = np.array([120.0 - land[0], 53.3 - land[1]], np.float32)
    dist_ball = np.linalg.norm(last_xy - land[None], axis=-1)
    tgt = last_xy[is_target.astype(bool)][0] if is_target.any() else land
    dist_tgt = np.linalg.norm(last_xy - tgt[None], axis=-1)
    static = np.stack([last_xy[:, 0] / 120.0, last_xy[:, 1] / 53.3,
                       dist_ball / 60.0, dist_tgt / 60.0,
                       side, is_target, is_passer], axis=-1).astype(np.float32)

    # ---- output phase ------------------------------------------------------
    O = min(int(out.frame_id.max()), cfg.max_output_frames)
    if O < cfg.min_output_frames:
        return None
    Y = np.zeros((P, O, 2), np.float32)
    valid = np.zeros((P, O), np.float32)
    for nfl_id, go in out.groupby("nfl_id", sort=False):
        i = pidx[nfl_id]
        f = go.frame_id.to_numpy() - 1
        m = f < O
        x = go.x.to_numpy(np.float32)[m]
        y = go.y.to_numpy(np.float32)[m]
        if flip:
            x, y = 120.0 - x, 53.3 - y
        Y[i, f[m], 0], Y[i, f[m], 1] = x, y
        valid[i, f[m]] = 1.0

    prev = np.concatenate([last_xy[:, None, :], Y[:, :-1]], axis=1)
    dY = (Y - prev).astype(np.float32)                     # delta targets

    return dict(dX=dX, X=X, s=SA, a=AA, o=OO, dir=DD, time_decay=time_decay,
                static=static, last_xy=last_xy.astype(np.float32), land=land,
                dY=dY, valid=valid, to_pred=to_pred, roles=role_codes(inp, players),
                S=np.int64(S), O=np.int64(O), P=np.int64(P))


class NFLDataset(Dataset):
    """Preprocessed plays of the given weeks (shared preparation cache)."""

    def __init__(self, plays: pd.DataFrame, cfg: Config, weeks: List[int], cache: Dict):
        self.items = []
        for w in weeks:
            for _, row in plays[plays.week == w].iterrows():
                key = (w, row.game_id, row.play_id)
                if key not in cache:
                    cache[key] = prepare_play(row.inp, row.out, cfg)
                prep = cache[key]
                if prep is not None:
                    self.items.append(prep)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        d = self.items[idx]
        return {k: torch.from_numpy(np.asarray(v)) for k, v in d.items()}


def collate_plays(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad players (P) and frames (S, O) to the batch maxima; build masks."""
    B = len(batch)
    P = max(int(b["P"]) for b in batch)
    S = max(int(b["S"]) for b in batch)
    O = max(int(b["O"]) for b in batch)

    def pad(t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        out = torch.zeros(shape, dtype=torch.float32)
        out[tuple(slice(0, d) for d in t.shape)] = t.float()
        return out

    out: Dict[str, torch.Tensor] = {
        "dX": torch.stack([pad(b["dX"], (P, S, 2)) for b in batch]),
        "X": torch.stack([pad(b["X"], (P, S, 2)) for b in batch]),
        "s": torch.stack([pad(b["s"], (P, S)) for b in batch]),
        "a": torch.stack([pad(b["a"], (P, S)) for b in batch]),
        "o": torch.stack([pad(b["o"], (P, S)) for b in batch]),
        "dir": torch.stack([pad(b["dir"], (P, S)) for b in batch]),
        "time_decay": torch.stack([pad(b["time_decay"], (S, 1)) for b in batch]),
        "static": torch.stack([pad(b["static"], (P, 7)) for b in batch]),
        "last_xy": torch.stack([pad(b["last_xy"], (P, 2)) for b in batch]),
        "land": torch.stack([b["land"].float() for b in batch]),
        "dY": torch.stack([pad(b["dY"], (P, O, 2)) for b in batch]),
        "valid": torch.stack([pad(b["valid"], (P, O)) for b in batch]),
        "to_pred": torch.stack([pad(b["to_pred"], (P,)) for b in batch]),
        "roles": torch.stack([pad(b["roles"], (P,)) for b in batch]).long(),
        "S": torch.tensor([int(b["S"]) for b in batch], dtype=torch.long),
        "O": torch.tensor([int(b["O"]) for b in batch], dtype=torch.long),
    }
    # existence masks: 1.0 where the player / frame really exists
    p_mask = torch.zeros(B, P)
    s_mask = torch.zeros(B, P, S)
    o_mask = torch.zeros(B, P, O)
    for i, b in enumerate(batch):
        Pi, Si, Oi = int(b["P"]), int(b["S"]), int(b["O"])
        p_mask[i, :Pi] = 1.0
        s_mask[i, :Pi, :Si] = 1.0
        o_mask[i, :Pi, :Oi] = 1.0
    out.update(p_mask=p_mask, s_mask=s_mask, o_mask=o_mask)
    return out


def make_loader(ds: Dataset, cfg: Config, shuffle: bool) -> DataLoader:
    """Deterministic loader: fixed generator seed, optional pin_memory."""
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                      num_workers=cfg.num_workers, collate_fn=collate_plays,
                      generator=g, persistent_workers=cfg.num_workers > 0,
                      pin_memory=torch.cuda.is_available())


def load_all_plays(cfg: Config) -> pd.DataFrame:
    """Load every week once and tag it with its split week number."""
    weeks = sorted(set(cfg.train_weeks) | set(cfg.val_weeks) | set(cfg.test_weeks))
    frames = []
    for w in weeks:
        inp, out = load_week(Path(cfg.data_dir), w)
        df = build_plays(inp, out)
        df["week"] = w
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Architecture
# --------------------------------------------------------------------------- #
def rope_angles(positions: torch.Tensor, dim: int, base: float) -> torch.Tensor:
    """RoPE rotation angles: positions (...) -> (..., dim // 2) [4]."""
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device,
                                       dtype=torch.float32) / dim))
    return positions[..., None].float() * inv[None, :]


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Rotate channel pairs of x (..., D) by angles (broadcastable, D//2)."""
    d = x.shape[-1]
    x1, x2 = x[..., 0:d:2], x[..., 1:d:2]
    cos, sin = torch.cos(angles), torch.sin(angles)
    out = torch.empty_like(x)
    out[..., 0:d:2] = x1 * cos - x2 * sin
    out[..., 1:d:2] = x1 * sin + x2 * cos
    return out


def _mask_zero(x: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
    """
    FIX-3: NaN-safe padding zeroing. REPLACE padded token embeddings with
    exact zeros. Multiplication (x * mask) cannot remove NaN (NaN * 0 = NaN),
    torch.where can.
    """
    return torch.where(key_pad.unsqueeze(-1), torch.zeros_like(x), x)


class DistanceBias(nn.Module):
    """Learned additive attention bias from pairwise distances (yards)."""

    def __init__(self, n_buckets: int, max_dist: float, n_heads: int):
        super().__init__()
        self.n_buckets, self.max_dist = n_buckets, max_dist
        self.emb = nn.Embedding(n_buckets, n_heads)
        nn.init.zeros_(self.emb.weight)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        b = (dist / self.max_dist * (self.n_buckets - 1)).clamp(0, self.n_buckets - 1).long()
        return self.emb(b).permute(0, 3, 1, 2)             # (B, H, P, P)


class MaskedAttention(nn.Module):
    """Manual MHA: key padding mask, 4-D additive bias, dead-row fix (FIX-3)."""

    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads, self.dh = n_heads, dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad=None, attn_bias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (B, L, self.n_heads, self.dh)
        q, k, v = [t.view(*shape).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if attn_bias is not None:
            scores = scores + attn_bias
        if key_pad is not None:
            scores = scores.masked_fill(key_pad[:, None, None, :], -torch.inf)
            # FIX-3: rows whose EVERY key is masked (padded player in temporal
            # attention, padded frame in spatial attention) would produce
            # softmax([-inf, ...]) = NaN; give them uniform scores instead.
            dead = key_pad.all(dim=-1)                     # (B,)
            if dead.any():
                scores = scores.masked_fill(dead[:, None, None, None], 0.0)
        attn = self.drop(torch.softmax(scores, dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, L, D))


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(dim * mult, dim))

    def forward(self, x):
        return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer layer; padded tokens replaced with zeros (FIX-3)."""

    def __init__(self, dim, n_heads, dropout, ffn_mult=4):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = MaskedAttention(dim, n_heads, dropout)
        self.ff = FeedForward(dim, ffn_mult, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad=None, attn_bias=None):
        x = x + self.drop(self.attn(self.ln1(x), key_pad, attn_bias))
        x = x + self.drop(self.ff(self.ln2(x)))
        if key_pad is not None:
            x = _mask_zero(x, key_pad)
        return x


class SqueezeFormerBlock(nn.Module):
    """FFN-Attention-Conv-FFN block for the temporal axis [5]; NaN-safe pads."""

    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(dim) for _ in range(4)])
        self.ff1 = FeedForward(dim, 2, dropout)
        self.attn = MaskedAttention(dim, n_heads, dropout)
        self.conv = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.GELU())
        self.ff2 = FeedForward(dim, 4, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad=None):
        x = x + 0.5 * self.drop(self.ff1(self.ln[0](x)))
        x = x + self.drop(self.attn(self.ln[1](x), key_pad))
        h = self.ln[2](x)
        if key_pad is not None:
            h = _mask_zero(h, key_pad)      # conv must not see NaN / garbage
        x = x + self.drop(self.conv(h.transpose(1, 2)).transpose(1, 2))
        x = x + 0.5 * self.drop(self.ff2(self.ln[3](x)))
        if key_pad is not None:
            x = _mask_zero(x, key_pad)
        return x


class SpatioTemporalBlockV1(nn.Module):
    """Temporal SqueezeFormer per player + distance-biased spatial attention."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.temporal = SqueezeFormerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.spatial = TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.dist_bias = DistanceBias(cfg.dist_bias_buckets, cfg.dist_bias_max, cfg.n_heads)

    def forward(self, x, s_mask, dist):
        B, P, S, D = x.shape
        xt = self.temporal(x.reshape(B * P, S, D), key_pad=(s_mask.reshape(B * P, S) < 0.5))
        xs = xt.reshape(B, P, S, D).permute(0, 2, 1, 3).reshape(B * S, P, D)
        pad_s = s_mask.permute(0, 2, 1).reshape(B * S, P) < 0.5
        bias = self.dist_bias(dist).unsqueeze(1).expand(B, S, -1, -1, -1).reshape(B * S, -1, P, P)
        xs = self.spatial(xs, key_pad=pad_s, attn_bias=bias)
        return xs.reshape(B, S, P, D).permute(0, 2, 1, 3)


class SpatioTemporalBlockV2(nn.Module):
    """Refinement over the output phase: horizons first, then players."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.temporal = TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.spatial = TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.dist_bias = DistanceBias(cfg.dist_bias_buckets, cfg.dist_bias_max, cfg.n_heads)

    def forward(self, x, o_mask, dist):
        B, P, O, D = x.shape
        xt = self.temporal(x.reshape(B * P, O, D), key_pad=(o_mask.reshape(B * P, O) < 0.5))
        xs = xt.reshape(B, P, O, D).permute(0, 2, 1, 3).reshape(B * O, P, D)
        pad_s = o_mask.permute(0, 2, 1).reshape(B * O, P) < 0.5
        bias = self.dist_bias(dist).unsqueeze(1).expand(B, O, -1, -1, -1).reshape(B * O, -1, P, P)
        xs = self.spatial(xs, key_pad=pad_s, attn_bias=bias)
        return xs.reshape(B, O, P, D).permute(0, 2, 1, 3)


class TrajectoryModel(nn.Module):
    """Encode pre-pass tracking -> per-(player, horizon) delta predictions."""

    STATIC_DIM = 7
    IN_FEAT_DIM = 9      # dX(2) + s + a + sin/cos(o) + sin/cos(dir) + time_decay
    OUT_FEAT_DIM = 4     # t_sec, progress, sin, cos of the horizon phase

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.temporal_proj = nn.Sequential(nn.Linear(self.IN_FEAT_DIM, d), nn.GELU(),
                                           nn.Linear(d, d))
        self.v1 = nn.ModuleList([SpatioTemporalBlockV1(cfg) for _ in range(cfg.n_temporal_blocks)])
        self.static_proj = nn.Linear(d + self.STATIC_DIM, d)
        self.spatial_pre = nn.ModuleList(
            [TransformerEncoderLayer(d, cfg.n_heads, cfg.dropout)
             for _ in range(cfg.n_spatial_layers_pre)])
        self.dist_bias_pre = DistanceBias(cfg.dist_bias_buckets, cfg.dist_bias_max, cfg.n_heads)
        self.out_feat_proj = nn.Linear(d + self.OUT_FEAT_DIM, d)
        self.bilstm = nn.LSTM(d, cfg.rnn_hidden, 2, batch_first=True,
                              bidirectional=True, dropout=cfg.dropout)
        self.bigru = nn.GRU(d, cfg.rnn_hidden, 2, batch_first=True,
                            bidirectional=True, dropout=cfg.dropout)
        self.mix_proj = nn.Linear(4 * cfg.rnn_hidden + d, d)
        self.v2 = nn.ModuleList([SpatioTemporalBlockV2(cfg) for _ in range(cfg.n_refine_blocks)])
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.cfg = cfg

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cfg = self.cfg
        B, P, S, _ = batch["X"].shape
        O = int(batch["dY"].shape[2])

        # input feature stack: deltas, speed, accel, angle sin/cos, recency
        feats = torch.cat([
            batch["dX"],
            batch["s"][..., None] / 10.0,
            batch["a"][..., None] / 20.0,
            torch.sin(torch.deg2rad(batch["o"]))[..., None],
            torch.cos(torch.deg2rad(batch["o"]))[..., None],
            torch.sin(torch.deg2rad(batch["dir"]))[..., None],
            torch.cos(torch.deg2rad(batch["dir"]))[..., None],
            batch["time_decay"].unsqueeze(1).expand(B, P, S, 1),
        ], dim=-1)
        h = self.temporal_proj(feats)                              # (B, P, S, D)

        dist = torch.cdist(batch["last_xy"], batch["last_xy"])     # (B, P, P)
        for blk in self.v1:
            h = blk(h, batch["s_mask"], dist)

        # pool the last VALID input frame per play (S varies inside a batch)
        idx = (batch["S"] - 1).clamp(min=0)[:, None, None, None].expand(B, P, 1, cfg.d_model)
        h_last = torch.gather(h, 2, idx).squeeze(2)                # (B, P, D)
        h = self.static_proj(torch.cat([h_last, batch["static"]], dim=-1))
        bias = self.dist_bias_pre(dist)
        for lyr in self.spatial_pre:
            h = lyr(h, key_pad=(batch["p_mask"] < 0.5), attn_bias=bias)

        # one token per (player, horizon); RoPE encodes the horizon index [4]
        horizon = torch.arange(O, device=h.device)
        angles = rope_angles(horizon, cfg.d_model, cfg.rope_base)  # (O, D//2)
        h = apply_rope(h.unsqueeze(2).expand(B, P, O, cfg.d_model),
                       angles[None, None])
        # FIX-2: every horizon feature must be expanded over the player axis
        t_sec = (horizon + 1).float() / 10.0                        # (O,)
        phase = torch.deg2rad(horizon * 36.0)                       # (O,)
        o_len = batch["O"][:, None, None, None].float().clamp(min=1.0)   # (B,1,1,1)
        prog = (horizon + 1).float()[None, None, :, None] / o_len   # (B,1,O,1)
        out_feats = torch.cat([
            t_sec[None, None, :, None].expand(B, P, O, 1),          # time, s
            prog.expand(B, P, O, 1),                                # o / O_i
            torch.sin(phase)[None, None, :, None].expand(B, P, O, 1),
            torch.cos(phase)[None, None, :, None].expand(B, P, O, 1),
        ], dim=-1)                                                  # (B,P,O,4)
        h = self.out_feat_proj(torch.cat([h, out_feats], dim=-1))   # (B,P,O,D)

        # recurrent mixing over horizons per player: BiLSTM || BiGRU + skip
        ht = h.reshape(B * P, O, cfg.d_model)
        lengths = batch["o_mask"].reshape(B * P, O).sum(-1).clamp(min=1).cpu().long()
        packed = nn.utils.rnn.pack_padded_sequence(ht, lengths, batch_first=True,
                                                   enforce_sorted=False)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(self.bilstm(packed)[0], batch_first=True)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(self.bigru(packed)[0], batch_first=True)
        h = self.mix_proj(torch.cat([lstm_out, gru_out, ht], dim=-1)).reshape(B, P, O, cfg.d_model)

        for blk in self.v2:
            h = blk(h, batch["o_mask"], dist)
        return self.head(h)                                        # (B, P, O, 2)


# --------------------------------------------------------------------------- #
# Optimizers, EMA, losses, official metric
# --------------------------------------------------------------------------- #
def _zeropower_via_newton_schulz(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate orthogonalization of a momentum matrix (Muon core) [3]."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    transposed = g.shape[-2] > g.shape[-1]
    x = g.mT if transposed else g
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = x @ x.mT
        x = a * x + b * (A @ x) + c * ((A @ A) @ x)
    return x.mT if transposed else x


class Muon(torch.optim.Optimizer):
    """Muon: momentum + Newton-Schulz orthogonalized updates for 2-D weights [3]."""

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "momentum" not in st:
                    st["momentum"] = torch.zeros_like(p.grad)
                st["momentum"].lerp_(p.grad, 1 - mu)
                update = _zeropower_via_newton_schulz(st["momentum"])
                scale = 0.2 * math.sqrt(max(p.shape[-2], p.shape[-1]))
                p.mul_(1 - lr * wd).add_(update, alpha=-lr * scale)


def build_optimizers(model: nn.Module, cfg: Config) -> List[torch.optim.Optimizer]:
    """
    FIX-4: returns a LIST of optimizers. By default (use_muon=False) everything
    is trained with AdamW (stable); with use_muon=True hidden 2-D matrices go
    to Muon and the rest stays on AdamW.
    """
    muon_p, adamw_p = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if cfg.use_muon and p.ndim >= 2 and not any(k in name for k in ("head", "emb", "ln")):
            muon_p.append(p)
        else:
            adamw_p.append(p)
    opts: List[torch.optim.Optimizer] = []
    if muon_p:
        opts.append(Muon(muon_p, lr=cfg.lr_muon, weight_decay=cfg.weight_decay))
    if adamw_p:
        opts.append(torch.optim.AdamW(adamw_p, lr=cfg.lr_adamw, weight_decay=cfg.weight_decay))
    return opts


class EMA:
    """Exponential moving average of weights; used for evaluation/saving."""

    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)


def huber_delta_loss(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber loss on per-frame deltas, averaged over valid (player, frame)."""
    loss = F.huber_loss(pred, target, reduction="none", delta=delta).sum(-1)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def official_rmse(err_sq_sum: float, n_points: int) -> float:
    """Competition metric: sqrt( 1/(2N) * sum(dx^2 + dy^2) )."""
    return float(math.sqrt(err_sq_sum / (2.0 * max(n_points, 1))))


def deltas_to_absolute(pred_delta: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """
    FIX-1: convert per-frame deltas to absolute coordinates.
    pred_delta: (..., O, 2); anchor: (..., 2) -> (..., 1, 2) so it broadcasts
    over the horizon axis of cumsum(..., axis=-2).
    """
    return anchor[..., None, :] + np.cumsum(pred_delta, axis=-2)