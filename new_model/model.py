import os, math, json, time, random, pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

# ============================ PATHS & SETUP ============================
WORK = Path(".") if Path("/kaggle").exists() else Path("./hastnet_out")
CKPT_DIR = WORK / "checkpoints"
FIG_DIR = WORK / "figures"
OUT_DIR = WORK / "outputs"
TUNE_DIR = WORK / "tuning"


def setup_dirs():
    for d in (
    CKPT_DIR, FIG_DIR, OUT_DIR, TUNE_DIR, TUNE_DIR / "checkpoints", TUNE_DIR / "figures", TUNE_DIR / "outputs"):
        d.mkdir(parents=True, exist_ok=True)


def find_data_dir() -> Path:
    for base in (Path("data"), Path(".")):
        hits = sorted(base.rglob("input_2023_w01.csv"))
        if hits: return hits[0].parent
    raise FileNotFoundError("input_2023_w01.csv not found")


DATA_DIR = find_data_dir()


def set_seed(seed: int) -> None:
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed);
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================ CONFIG ============================
@dataclass
class Config:
    train_weeks: list = field(default_factory=lambda: list(range(1, 16)))
    val_weeks: list = field(default_factory=lambda: [16, 17])
    test_weeks: list = field(default_factory=lambda: [18])
    max_input_frames: int = 80
    max_output_frames: int = 96
    min_output_frames: int = 5
    d_model: int = 128
    n_heads: int = 8
    n_v1_blocks: int = 4
    n_pre_spatial: int = 2
    n_v2_blocks: int = 2
    n_control: int = 8
    rnn_hidden: int = 96
    dropout: float = 0.1
    dist_buckets: int = 16
    dist_max: float = 60.0
    fourier_freqs: int = 4
    seed: int = 42
    epochs: int = 25
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    huber_delta: float = 0.35
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    patience: int = 6
    amp: bool = False
    use_muon: bool = False
    w_pos: float = 1.0
    w_end: float = 0.5
    w_delta: float = 0.5
    w_acc: float = 0.1
    w_nll: float = 0.1
    p_flip: float = 0.5
    p_rot: float = 0.5
    rot_deg: float = 10.0
    p_shift: float = 0.5
    shift_yd: float = 3.0
    p_fshift: float = 0.3
    n_bootstrap: int = 1000
    bootstrap_block: int = 8
    num_workers: int = 2
    exp_name: str = "hastnet"


# ============================ DATA PREPARATION ============================
FIELD_C = np.array([60.0, 26.65], np.float32)
ROLE_LIST = ["Defensive Coverage", "Targeted Receiver", "Passer", "Other Route Runner"]


def _norm_angle(a): return (a + 180.0) % 360.0 - 180.0


def build_plays(inp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    inp = inp.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    out = out.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    g_out = out.groupby(["game_id", "play_id"], sort=False)
    rows = []
    for (gid, pid), gi in inp.groupby(["game_id", "play_id"], sort=False):
        try:
            go = g_out.get_group((gid, pid))
        except KeyError:
            continue
        f0 = gi.iloc[0]
        rows.append(dict(game_id=gid, play_id=pid, inp=gi, out=go,
                         land=np.array([f0.ball_land_x, f0.ball_land_y], np.float32)))
    return pd.DataFrame(rows)


def prepare_play(inp, out, cfg) -> dict:
    players = inp.nfl_id.unique();
    P = len(players)
    pidx = {p: i for i, p in enumerate(players)}
    flip = inp.play_direction.iloc[0] == "left"
    ash = 180.0 if flip else 0.0
    S = min(int(inp.frame_id.max()), cfg.max_input_frames)
    inp = inp[inp.frame_id <= S]
    X = np.zeros((P, S, 2), np.float32);
    FT = np.zeros((P, S, 9), np.float32)
    last_delta = np.zeros((P, 2), np.float32)
    static = np.zeros((P, 7), np.float32);
    to_pred = np.zeros(P, np.float32)
    roles = np.zeros(P, np.int64)
    for nfl_id, gi in inp.groupby("nfl_id", sort=False):
        i = pidx[nfl_id];
        fr = np.clip(gi.frame_id.to_numpy() - 1, 0, S - 1)
        x = gi.x.to_numpy(np.float32);
        y = gi.y.to_numpy(np.float32)
        if flip: x, y = 120.0 - x, 53.3 - y
        X[i, fr, 0], X[i, fr, 1] = x, y
        dx = np.zeros_like(x);
        dx[1:] = x[1:] - x[:-1]
        dy = np.zeros_like(y);
        dy[1:] = y[1:] - y[:-1]
        d = _norm_angle(gi.dir.to_numpy(np.float32) + ash)
        o = _norm_angle(gi.o.to_numpy(np.float32) + ash)
        FT[i, fr, 0], FT[i, fr, 1] = dx, dy
        FT[i, fr, 2], FT[i, fr, 3] = gi.s.to_numpy(np.float32) / 10.0, gi.a.to_numpy(np.float32) / 20.0
        FT[i, fr, 4], FT[i, fr, 5] = np.sin(np.deg2rad(d)), np.cos(np.deg2rad(d))
        FT[i, fr, 6], FT[i, fr, 7] = np.sin(np.deg2rad(o)), np.cos(np.deg2rad(o))
        last_delta[i] = np.array([dx[-1], dy[-1]], np.float32)
        to_pred[i] = float(gi.player_to_predict.iloc[0])
        roles[i] = ROLE_LIST.index(gi.player_role.iloc[0]) if gi.player_role.iloc[0] in ROLE_LIST else 3
        static[i, 4] = float(gi.player_side.iloc[0] == "Offense")
        static[i, 5], static[i, 6] = float(gi.player_role.iloc[0] == "Targeted Receiver"), float(
            gi.player_role.iloc[0] == "Passer")
    td = np.exp(-(S - 1 - np.arange(S, dtype=np.float32)) / 10.0)[:, None]
    FT[:, :, 8:9] = td[None]
    last_xy = X[:, -1].copy()
    land = inp.iloc[0][["ball_land_x", "ball_land_y"]].to_numpy(np.float32).copy()
    if flip: land = np.array([120.0 - land[0], 53.3 - land[1]], np.float32)
    tgt = last_xy[roles == 1][0] if (roles == 1).any() else land
    static[:, 0], static[:, 1] = last_xy[:, 0] / 120.0, last_xy[:, 1] / 53.3
    static[:, 2] = np.linalg.norm(last_xy - land[None], axis=-1) / 60.0
    static[:, 3] = np.linalg.norm(last_xy - tgt[None], axis=-1) / 60.0
    O = min(int(out.frame_id.max()), cfg.max_output_frames)
    if O < cfg.min_output_frames: return None
    pos = np.zeros((P, O, 2), np.float32);
    valid = np.zeros((P, O), np.float32)
    for nfl_id, go in out.groupby("nfl_id", sort=False):
        i = pidx[nfl_id];
        fr = go.frame_id.to_numpy() - 1;
        m = fr < O
        x = go.x.to_numpy(np.float32)[m];
        y = go.y.to_numpy(np.float32)[m]
        if flip: x, y = 120.0 - x, 53.3 - y
        pos[i, fr[m], 0], pos[i, fr[m], 1] = x, y
        valid[i, fr[m]] = 1.0
    prev = np.concatenate([last_xy[:, None], pos[:, :-1]], 1)
    delta = pos - prev
    accel = delta - np.concatenate([last_delta[:, None], delta[:, :-1]], 1)
    endpoint = np.zeros((P, 2), np.float32)
    for i in range(P):
        li = int(valid[i].sum()) - 1
        endpoint[i] = pos[i, max(li, 0)]
    return dict(X=X, feat=FT, static=static, last_xy=last_xy, last_delta=last_delta,
                pos=pos, delta=delta, accel=accel, endpoint=endpoint, valid=valid,
                to_pred=to_pred, roles=roles, S_len=np.int64(S), O_len=np.int64(O))


def load_and_prepare_data(cfg, smoke_test=False):
    cache_name = "prep_cache_smoke.pkl" if smoke_test else "prep_cache.pkl"
    CACHE = OUT_DIR / cache_name
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            PREP = pickle.load(f)
        print("prep cache loaded:", len(PREP), "plays")
    else:
        PREP = {}
        weeks = [1] if smoke_test else sorted(set(cfg.train_weeks) | set(cfg.val_weeks) | set(cfg.test_weeks))
        for w in tqdm(weeks, desc="prepare weeks"):
            inp = pd.read_csv(DATA_DIR / f"input_2023_w{w:02d}.csv")
            out = pd.read_csv(DATA_DIR / f"output_2023_w{w:02d}.csv")
            plays_df = build_plays(inp, out)
            if smoke_test: plays_df = plays_df.head(10)
            for _, r in plays_df.iterrows():
                PREP[(w, r.game_id, r.play_id)] = prepare_play(r.inp, r.out, cfg)
        with open(CACHE, "wb") as f:
            pickle.dump(PREP, f)
        print("prepared:", len(PREP), "plays")

    idx_df = pd.DataFrame([(w, g, p) for (w, g, p) in PREP.keys()], columns=["week", "game_id", "play_id"])
    if smoke_test:
        idx_df = idx_df.head(10)
        keys = set(zip(idx_df.week, idx_df.game_id, idx_df.play_id))
        PREP = {k: v for k, v in PREP.items() if k in keys}
    return PREP, idx_df


# ============================ DATASET & LOADER ============================
class HASTDataset(Dataset):
    def __init__(self, plays, cfg, weeks, prep, train=False):
        self.cfg, self.train = cfg, train
        self.items = [prep[(w, r.game_id, r.play_id)]
                      for w in weeks for _, r in plays[plays.week == w].iterrows()
                      if prep.get((w, r.game_id, r.play_id)) is not None]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        X, feat, pos = d["X"].copy(), d["feat"].copy(), d["pos"].copy()
        delta, accel, last = d["delta"].copy(), d["accel"].copy(), d["last_xy"].copy()
        ldel, end = d["last_delta"].copy(), d["endpoint"].copy()
        S, O = int(d["S_len"]), int(d["O_len"])

        if self.train and self.cfg.p_fshift > 0:
            k = int(np.random.randint(0, 4))
            if k > 0 and S - k >= 8:
                X, feat = X[:, k:].copy(), feat[:, k:].copy()
                S = S - k
                feat[:, 0, 0:2] = 0.0

        td = np.exp(-(S - 1 - np.arange(S, dtype=np.float32)) / 10.0)
        feat[:, :, 8:9] = td[None, :, None]
        to_pred, valid = d.get("to_pred"), d.get("valid")
        if to_pred is None: to_pred = (np.abs(d["endpoint"]).sum(-1) > 0).astype(np.float32)
        if valid is None: valid = np.ones((to_pred.shape[0], O), np.float32) * to_pred[:, None]

        return {"X": torch.from_numpy(X), "feat": torch.from_numpy(feat), "pos": torch.from_numpy(pos),
                "delta": torch.from_numpy(delta), "accel": torch.from_numpy(accel),
                "last_xy": torch.from_numpy(last), "last_delta": torch.from_numpy(ldel),
                "endpoint": torch.from_numpy(end), "static": torch.from_numpy(d["static"]),
                "roles": torch.from_numpy(d["roles"]), "valid": torch.from_numpy(valid),
                "to_pred": torch.from_numpy(to_pred), "S_len": torch.tensor(S), "O_len": torch.tensor(O)}


def collate(batch):
    B = len(batch)
    P = max(int(b["roles"].numel()) for b in batch)
    S = max(int(b["S_len"]) for b in batch)
    O = max(int(b["O_len"]) for b in batch)

    def pad(t, shape):
        o = torch.zeros(shape, dtype=torch.float32)
        o[tuple(slice(0, d) for d in t.shape)] = t.float()
        return o

    out = {
        "X": torch.stack([pad(b["X"], (P, S, 2)) for b in batch]),
        "feat": torch.stack([pad(b["feat"], (P, S, 9)) for b in batch]),
        "pos": torch.stack([pad(b["pos"], (P, O, 2)) for b in batch]),
        "delta": torch.stack([pad(b["delta"], (P, O, 2)) for b in batch]),
        "accel": torch.stack([pad(b["accel"], (P, O, 2)) for b in batch]),
        "last_xy": torch.stack([pad(b["last_xy"], (P, 2)) for b in batch]),
        "last_delta": torch.stack([pad(b["last_delta"], (P, 2)) for b in batch]),
        "endpoint": torch.stack([pad(b["endpoint"], (P, 2)) for b in batch]),
        "static": torch.stack([pad(b["static"], (P, 7)) for b in batch]),
        "roles": torch.stack([pad(b["roles"], (P,)) for b in batch]).long(),
        "S_len": torch.tensor([int(b["S_len"]) for b in batch]),
        "O_len": torch.tensor([int(b["O_len"]) for b in batch]),
        "valid": torch.stack([pad(b["valid"], (P, O)) for b in batch]),
        "to_pred": torch.stack([pad(b["to_pred"], (P,)) for b in batch])
    }
    pm, sm, om = torch.zeros(B, P), torch.zeros(B, P, S), torch.zeros(B, P, O)
    for i, b in enumerate(batch):
        Pi, Si, Oi = int(b["roles"].numel()), int(b["S_len"]), int(b["O_len"])
        pm[i, :Pi], sm[i, :Pi, :Si], om[i, :Pi, :Oi] = 1.0, 1.0, 1.0
    out.update(p_mask=pm, s_mask=sm, o_mask=om)
    return out


def worker_init(worker_id): np.random.seed(42 + worker_id)


def make_loader(ds, shuffle, cfg):
    g = torch.Generator();
    g.manual_seed(cfg.seed)
    return DataLoader(ds, cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
                      collate_fn=collate, generator=g, worker_init_fn=worker_init,
                      persistent_workers=cfg.num_workers > 0)


# ============================ MODEL COMPONENTS ============================
def _mask_zero(x, pad): return torch.where(pad.unsqueeze(-1), torch.zeros_like(x), x)


def fourier(t, nf):
    f = 2.0 ** torch.arange(nf, device=t.device, dtype=torch.float32)
    a = 2 * math.pi * t[..., None] * f[None, :] if t.dim() else 2 * math.pi * t[:, None] * f[None, :]
    return torch.cat([torch.sin(a), torch.cos(a)], -1)


class MaskedAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__();
        self.h, self.dh = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim);
        self.o = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_pad=None, bias=None):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [t.view(B, L, self.h, self.dh).transpose(1, 2) for t in (q, k, v)]
        s = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        if bias is not None: s = s + bias
        if key_pad is not None:
            s = s.masked_fill(key_pad[:, None, None, :], -torch.inf)
            dead = key_pad.all(-1)
            if dead.any(): s = s.masked_fill(dead.view(B, 1, 1, 1), 0.0)
        attn = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(self.drop(attn))


class CrossAttention(nn.Module):
    def __init__(self, qd, kd, heads, dropout):
        super().__init__()
        self.h, self.dh = heads, qd // heads
        self.q, self.k, self.v, self.o = nn.Linear(qd, qd), nn.Linear(kd, qd), nn.Linear(kd, qd), nn.Linear(qd, qd)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, kv, key_pad):
        B, Lq, _ = q.shape;
        Lk = kv.shape[1]
        qh = self.q(q).view(B, Lq, self.h, self.dh).transpose(1, 2)
        kh = self.k(kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        vh = self.v(kv).view(B, Lk, self.h, self.dh).transpose(1, 2)
        s = qh @ kh.transpose(-1, -2) / math.sqrt(self.dh)
        if key_pad is not None:
            s = s.masked_fill(key_pad[:, None, None, :], -torch.inf)
            dead = key_pad.all(-1)
            if dead.any(): s = s.masked_fill(dead[:, None, None, None], 0.0)
        out = (s.softmax(-1) @ vh).transpose(1, 2).reshape(B, Lq, -1)
        return self.o(self.drop(out))


class FF(nn.Module):
    def __init__(self, d, m=4, drop=0.1):
        super().__init__();
        self.n = nn.Sequential(nn.Linear(d, d * m), nn.GELU(), nn.Dropout(drop), nn.Linear(d * m, d))

    def forward(self, x): return self.n(x)


class DistBias(nn.Module):
    def __init__(self, nb, dmax, heads):
        super().__init__();
        self.nb, self.dmax = nb, dmax
        self.e = nn.Embedding(nb, heads);
        nn.init.zeros_(self.e.weight)

    def forward(self, d):
        b = (d / self.dmax * (self.nb - 1)).clamp(0, self.nb - 1).long()
        return self.e(b).permute(0, 3, 1, 2)


class SqueezeFormerBlock(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.f1, self.f2 = FF(d, 2, drop), FF(d, 4, drop)
        self.attn = MaskedAttention(d, h, drop)
        self.conv = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.GELU(),
                                  nn.Conv1d(d, d, 3, padding=1, groups=d), nn.GELU())
        self.drop = nn.Dropout(drop)

    def forward(self, x, kp):
        x = x + 0.5 * self.drop(self.f1(self.ln[0](x)))
        x = x + self.drop(self.attn(self.ln[1](x), kp))
        h = _mask_zero(self.ln[2](x), kp) if kp is not None else self.ln[2](x)
        x = x + self.drop(self.conv(h.transpose(1, 2)).transpose(1, 2))
        x = x + 0.5 * self.drop(self.f2(self.ln[3](x)))
        return _mask_zero(x, kp) if kp is not None else x


class STBlockV1Bias(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.t = SqueezeFormerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln = nn.LayerNorm(cfg.d_model);
        self.ff = FF(cfg.d_model, 4, cfg.dropout)
        self.attn = MaskedAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.bias = DistBias(cfg.dist_buckets, cfg.dist_max, cfg.n_heads)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, s_mask, p_mask, dist):
        B, P, S, D = x.shape
        x = self.t(x.reshape(B * P, S, D), (s_mask.reshape(B * P, S) < .5))
        xs = x.reshape(B, P, S, D).permute(0, 2, 1, 3).reshape(B * S, P, D)
        kp = s_mask.permute(0, 2, 1).reshape(B * S, P) < .5
        bias = self.bias(dist).unsqueeze(1).expand(B, S, -1, -1, -1).reshape(B * S, -1, P, P)
        xs = xs + self.drop(self.attn(self.ln(xs), kp, bias))
        xs = xs + self.drop(self.ff(self.ln(xs)))
        xs = _mask_zero(xs, kp)
        return xs.reshape(B, S, P, D).permute(0, 2, 1, 3)


class HASTNet(nn.Module):
    def __init__(self, cfg):
        super().__init__();
        self.cfg = cfg;
        d = cfg.d_model
        self.stem = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.v1 = nn.ModuleList([STBlockV1Bias(cfg) for _ in range(cfg.n_v1_blocks)])
        self.gq = nn.Parameter(torch.zeros(1, 1, d));
        nn.init.normal_(self.gq, std=0.02)
        self.gattn = CrossAttention(d, d, cfg.n_heads, cfg.dropout)
        self.gln = nn.LayerNorm(d)
        self.zproj = nn.Linear(d + 7, d)
        self.pre = nn.ModuleList([nn.TransformerEncoderLayer(d, cfg.n_heads, d * 4, cfg.dropout,
                                                             batch_first=True, norm_first=True)
                                  for _ in range(cfg.n_pre_spatial)])
        self.end_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.tau_h = nn.Linear(d, 1);
        self.w_h = nn.Linear(d, 1)
        M, nf = cfg.n_control, cfg.fourier_freqs
        self.qproj = nn.Linear(2 * nf, d)
        self.cross = CrossAttention(d, d, cfg.n_heads, cfg.dropout)
        self.cgru = nn.GRU(d, cfg.rnn_hidden, 2, batch_first=True, bidirectional=True, dropout=cfg.dropout)
        self.r_head = nn.Linear(2 * cfg.rnn_hidden, 2)
        E = 2 * nf
        self.hemb = nn.Sequential(nn.Linear(E, d), nn.GELU())
        self.refine = nn.Sequential(nn.Linear(2 + d + d, d), nn.GELU(), nn.Dropout(cfg.dropout))
        self.rho = nn.Linear(d, 2)
        self.lvar = nn.Linear(d, 2)
        self.M, self.nf = M, nf

    def forward(self, b):
        cfg = self.cfg;
        B, P, S, _ = b["X"].shape;
        O = int(b["pos"].shape[2]);
        d = cfg.d_model
        h = self.stem(b["feat"])
        dist = torch.cdist(b["last_xy"], b["last_xy"])
        for blk in self.v1: h = blk(h, b["s_mask"], b["p_mask"], dist)
        gs = h.reshape(B, P, S, d).permute(0, 2, 1, 3).reshape(B * S, P, d)
        g = self.gattn(self.gq.expand(B * S, 1, d), self.gln(gs),
                       (b["s_mask"].permute(0, 2, 1).reshape(B * S, P) < .5)).reshape(B, S, d)
        idx = (b["S_len"] - 1).clamp(min=0)
        h_last = h.gather(2, idx.view(B, 1, 1, 1).expand(B, P, 1, d)).squeeze(2)
        g = g.gather(1, idx.view(B, 1, 1).expand(B, 1, d)).squeeze(1)
        z = self.zproj(torch.cat([h_last, b["static"]], -1))
        for lyr in self.pre: z = lyr(z, src_key_padding_mask=(b["p_mask"] < .5))
        last = b["last_xy"]
        end = last + self.end_head(z)
        tau = 0.1 + 0.8 * torch.sigmoid(self.tau_h(z)).squeeze(-1)
        w = 0.05 + F.softplus(self.w_h(z)).squeeze(-1)
        tf = (torch.arange(O, device=z.device).float() + 1) / b["O_len"].clamp(min=1)[:, None].float()
        sp = torch.sigmoid((tf[:, None, :] - tau[:, :, None]) / w[:, :, None])
        last4 = last[:, :, None, :]
        anchor = last4 + (end[:, :, None, :] - last4) * sp[..., None]
        fm = (torch.arange(self.M, device=z.device).float() + 1) / self.M
        q = self.qproj(fourier(fm, self.nf))[None, None] + z[:, :, None, :]
        kv = torch.cat([z, g.unsqueeze(1)], 1)
        kp = torch.cat([b["p_mask"] < .5, torch.zeros(B, 1, dtype=torch.bool, device=z.device)], 1)
        q = self.cross(q.reshape(B * P, self.M, d), kv.repeat_interleave(P, 0), kp.repeat_interleave(P, 0))
        r = self.r_head(self.cgru(q)[0]).reshape(B, P, self.M, 2)
        pf = tf * self.M - 1.0
        ci = pf.floor().long().clamp(0, self.M - 2)
        wgt = (pf - ci.float()).clamp(0, 1)
        ig = ci.view(B, 1, O, 1).expand(B, P, O, 2)
        interp = r.gather(2, ig) * (1 - wgt)[:, None, :, None] + r.gather(2, ig + 1) * wgt[:, None, :, None]
        he = self.hemb(fourier(tf, self.nf))[:, None].expand(B, P, O, -1)
        hid = self.refine(torch.cat([interp, he, z[:, :, None, :].expand(B, P, O, d)], -1))
        pos = anchor + self.rho(hid)
        logvar = self.lvar(hid).clamp(-6, 6)
        return dict(pos=pos, logvar=logvar, end=end)


# ============================ LOSSES & EVALUATION ============================
def masked_huber(pred, tgt, mask, delta):
    e = F.huber_loss(pred, tgt, reduction="none", delta=delta).sum(-1)
    return (e * mask).sum() / mask.sum().clamp(min=1)


def hast_loss(out, b, cfg):
    mask = b["valid"] * b["to_pred"][..., None]
    O = out["pos"].shape[2]
    L_pos = masked_huber(out["pos"], b["pos"], mask, cfg.huber_delta)
    pm = b["to_pred"] * b["p_mask"]
    L_end = (F.huber_loss(out["end"], b["endpoint"], reduction="none", delta=cfg.huber_delta).sum(
        -1) * pm).sum() / pm.sum().clamp(min=1)
    d_pred = out["pos"] - torch.cat([b["last_xy"][:, :, None], out["pos"][:, :, :-1]], 2)
    dec = torch.exp(-0.03 * torch.arange(O, device=out["pos"].device)).float()
    L_del = masked_huber(d_pred, b["delta"], mask * dec[None, None, :], cfg.huber_delta)
    a_pred = d_pred - torch.cat([b["last_delta"][:, :, None], d_pred[:, :, :-1]], 2)
    L_acc = masked_huber(a_pred, b["accel"], mask, cfg.huber_delta)
    err2 = (out["pos"] - b["pos"]) ** 2
    nll = 0.5 * (out["logvar"] + err2 / torch.exp(out["logvar"]).clamp(min=1e-6))
    L_nll = (nll.sum(-1) * mask).sum() / mask.sum().clamp(min=1)
    total = cfg.w_pos * L_pos + cfg.w_end * L_end + cfg.w_delta * L_del + cfg.w_acc * L_acc + cfg.w_nll * L_nll
    return total, dict(pos=L_pos.item(), end=L_end.item(), delta=L_del.item(), acc=L_acc.item(), nll=L_nll.item())


class MuonLike(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p.grad))
                m.lerp_(p.grad, 0.05)
                x = m / (m.norm() + 1e-7)
                for _ in range(5):
                    A = x @ x.T;
                    x = 3.4445 * x - 4.7750 * (A @ x) + 2.0315 * (A @ A @ x)
                p.add_(x, alpha=-g["lr"])


def build_optimizers(model, cfg):
    if cfg.use_muon:
        from torch.optim import AdamW
        m2 = [p for n, p in model.named_parameters() if p.ndim >= 2 and "head" not in n]
        rest = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "head" not in n)]
        return [MuonLike(m2, cfg.lr), AdamW(rest, lr=cfg.lr, weight_decay=cfg.weight_decay)]
    return [torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)]


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()


@contextmanager
def ema_weights(model, ema):
    live = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema.shadow)
    try:
        yield
    finally:
        model.load_state_dict(live)


def official_rmse(sq_sum, n): return float(math.sqrt(sq_sum / (2.0 * max(n, 1))))


@torch.no_grad()
def predict_pos(model, b):
    out = model(b)
    return out, out["pos"]


@torch.no_grad()
def eval_rmse(model, loader, cfg, device):
    model.eval();
    sq = n = 0
    for b in loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        _, pos = predict_pos(model, b)
        m = (b["valid"] * b["to_pred"][..., None]).bool()
        sq += float(((pos - b["pos"]) ** 2).sum(-1)[m].sum());
        n += int(m.sum())
    return official_rmse(sq, n)


@torch.no_grad()
def collect_test_stats(model, loader, cfg, device):
    model.eval()
    plays_sq, plays_n, hor_sq, hor_n, role_sq, role_n, sig_pred, sig_obs = [], [], [], [], [], [], [], []
    for b in loader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
        out = model(b);
        pos = out["pos"]
        m = (b["valid"] * b["to_pred"][..., None]).bool()
        m_np = m.cpu().numpy()
        e2_np = ((pos - b["pos"]) ** 2).sum(-1).cpu().numpy()
        sig_pred.append(np.sqrt(np.exp(out["logvar"].cpu().numpy()).mean(-1))[m_np])
        sig_obs.append(np.sqrt(e2_np)[m_np])
        for i in range(e2_np.shape[0]):
            Pi = int(b["p_mask"][i].sum());
            Oi = int(b["o_mask"][i, 0].sum())
            mi = m[i, :Pi, :Oi].cpu().numpy()
            plays_sq.append(float(e2_np[i, :Pi, :Oi][mi].sum()));
            plays_n.append(int(mi.sum()))
            for o in range(Oi):
                mo = mi[:, o]
                if mo.any(): hor_sq.append(float(e2_np[i, :Pi, o][mo].sum())); hor_n.append(int(mo.sum()))
            roles = b["roles"][i, :Pi].cpu().numpy()
            for p in range(Pi):
                mp = mi[p]
                if mp.any(): role_sq.append(float(e2_np[i, p, :Oi][mp].sum())); role_n.append(int(mp.sum()))
    return dict(plays_sq=np.array(plays_sq), plays_n=np.array(plays_n), hor_sq=np.array(hor_sq), hor_n=np.array(hor_n),
                role_sq=np.array(role_sq), role_n=np.array(role_n), sig_pred=np.concatenate(sig_pred),
                sig_obs=np.concatenate(sig_obs))


def bootstrap_rmse(sq, n, rng, n_boot, block):
    point = official_rmse(sq.sum(), n.sum());
    N = len(sq)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s, min(s + block, N)) for s in rng.integers(0, N, max(1, N // block))])
        idx = idx[idx < N]
        boots.append(official_rmse(sq[idx].sum(), n[idx].sum()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


@torch.no_grad()
def const_velocity_stats(loader, cfg):
    sq, n = [], []
    for b in loader:
        last, ldel = b["last_xy"], b["last_delta"]
        O = int(b["pos"].shape[2])
        t = (torch.arange(O).float() + 1)[None, None, :, None]
        pos = last[:, :, None, :] + ldel[:, :, None, :] * t
        m = (b["valid"] * b["to_pred"][..., None]).bool()
        e2 = ((pos - b["pos"]) ** 2).sum(-1)
        for i in range(e2.shape[0]):
            Pi = int(b["p_mask"][i].sum());
            Oi = int(b["o_mask"][i, 0].sum());
            mi = m[i, :Pi, :Oi]
            sq.append(float(e2[i, :Pi, :Oi][mi].sum()));
            n.append(int(mi.sum()))
    return np.array(sq), np.array(n)


def per_horizon_rmse_correct(model, loader, device, maxO=48, n_boot=200, seed=0):
    model.eval()
    sq = [[] for _ in range(maxO)]
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            out = model(b)
            m = (b["valid"] * b["to_pred"][..., None]).bool().cpu().numpy()
            e2 = ((out["pos"] - b["pos"]) ** 2).sum(-1).cpu().numpy()
            for o in range(min(maxO, e2.shape[2])):
                v = e2[:, :, o][m[:, :, o]]
                if v.size: sq[o].append(v)
    rmse, lo, hi = [], [], []
    for o in range(maxO):
        if not sq[o]: rmse.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
        v = np.concatenate(sq[o])
        pt = float(np.sqrt(v.mean() / 2.0))
        rmse.append(pt)
        boots = [np.sqrt(v[rng.integers(0, v.size, v.size)].mean() / 2.0) for _ in range(n_boot)]
        lo.append(np.percentile(boots, 2.5));
        hi.append(np.percentile(boots, 97.5))
    return np.array(rmse), np.array(lo), np.array(hi)


# ============================ TRAINING LOOP ============================
def train_hastnet(cfg, prep, idx_df, device, out_dir, ckpt_dir):
    set_seed(cfg.seed)
    tr = HASTDataset(idx_df, cfg, cfg.train_weeks, prep, train=True)
    va = HASTDataset(idx_df, cfg, cfg.val_weeks, prep, train=False)
    te = HASTDataset(idx_df, cfg, cfg.test_weeks, prep, train=False)
    tr_loader, va_loader, te_loader = make_loader(tr, True, cfg), make_loader(va, False, cfg), make_loader(te, False,
                                                                                                           cfg)

    model = HASTNet(cfg).to(device)
    print(f"[train] params={sum(p.numel() for p in model.parameters()):,} train={len(tr)} val={len(va)} test={len(te)}")
    opts = build_optimizers(model, cfg)
    total = max(1, len(tr_loader) * cfg.epochs)
    lam = lambda s: (s / cfg.warmup_steps if s < cfg.warmup_steps else
                     max(0.0,
                         0.5 * (1 + math.cos(math.pi * (s - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)))))
    scheds = [LambdaLR(o, lam) for o in opts]
    ema = EMA(model, cfg.ema_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    ac = torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg.amp)
    history = dict(train_loss=[], val_rmse=[], parts=[])
    best, bad = math.inf, 0

    for ep in range(cfg.epochs):
        model.train();
        lsum = msum = 0.0;
        parts = []
        pbar = tqdm(tr_loader, desc=f"epoch {ep:02d}", leave=False)
        for b in pbar:
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            for o in opts: o.zero_grad(set_to_none=True)
            with ac:
                out = model(b)
                loss, parts_d = hast_loss(out, b, cfg)
            scaler.scale(loss).backward()
            for o in opts:
                scaler.unscale_(o)
                nn.utils.clip_grad_norm_([p for g in o.param_groups for p in g["params"]], cfg.grad_clip)
            for o in opts: scaler.step(o)
            scaler.update()
            for s in scheds: s.step()
            ema.update(model)
            m = float((b["valid"] * b["to_pred"][..., None]).sum())
            lsum += loss.item() * m;
            msum += m;
            parts.append(parts_d)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        with ema_weights(model, ema):
            vrmse = eval_rmse(model, va_loader, cfg, device)
        history["train_loss"].append(lsum / max(1, msum));
        history["val_rmse"].append(vrmse)
        history["parts"].append({k: float(np.mean([p[k] for p in parts])) for k in parts[0]})
        print(f"epoch {ep:02d} | train_loss={history['train_loss'][-1]:.4f} | val_rmse={vrmse:.4f}")
        if vrmse < best - 1e-4:
            best, bad = vrmse, 0
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model=model.state_dict(), ema=ema.shadow, config=asdict(cfg),
                            epoch=ep, val_rmse=vrmse), ckpt_dir / f"{cfg.exp_name}_best.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"[train] early stop @ epoch {ep} (best {best:.4f})");
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    return model, ema, history, te_loader