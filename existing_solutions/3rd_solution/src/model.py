from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x, key_padding_mask=None):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


class STTransformer(nn.Module):
    """Transparent educational reproduction of the published 3rd-place core."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 192,
        heads: int = 6,
        layers: int = 2,
        max_time: int = 20,
        max_players: int = 22,
        horizon: int = 20,
    ):
        super().__init__()
        self.hidden = hidden
        self.horizon = horizon

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.time_emb = nn.Parameter(torch.randn(1, max_time, 1, hidden) * 0.02)
        self.player_emb = nn.Parameter(torch.randn(1, 1, max_players, hidden) * 0.02)

        # Main path: per-player temporal modeling.
        self.temporal = nn.ModuleList(
            [TransformerBlock(hidden, heads) for _ in range(layers)]
        )
        # Interaction path: player-player attention at every observed frame.
        self.spatial = nn.ModuleList(
            [TransformerBlock(hidden, heads) for _ in range(layers)]
        )

        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
        )

        # Main future trajectory head: direct displacement from final input frame.
        self.motion_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, horizon * 2)
        )
        # Auxiliary heads from the write-up's multi-task idea.
        self.inter_head = nn.Linear(hidden, 2)      # input t -> input t+1
        self.endpoint_head = nn.Linear(hidden, 2)   # input t -> final future point

    def forward(self, x, time_mask, player_mask):
        # x: [B,T,P,F]
        B, T, P, _ = x.shape
        h = self.input_proj(x)
        h = h + self.time_emb[:, :T] + self.player_emb[:, :, :P]
        h = h * player_mask[:, None, :, None]

        # Temporal attention: each player looks over its own history.
        z = h.permute(0, 2, 1, 3).reshape(B * P, T, self.hidden)
        tmask = (~time_mask.bool()).unsqueeze(1).expand(B, P, T).reshape(B * P, T)
        for block in self.temporal:
            z = block(z, key_padding_mask=tmask)
        tfeat = z.reshape(B, P, T, self.hidden).permute(0, 2, 1, 3)
        tfeat = tfeat * time_mask[:, :, None, None] * player_mask[:, None, :, None]

        # Spatial attention: at each time, players attend to other players.
        z = tfeat.reshape(B * T, P, self.hidden)
        pmask = (~player_mask.bool()).unsqueeze(1).expand(B, T, P).reshape(B * T, P)
        for block in self.spatial:
            z = block(z, key_padding_mask=pmask)
        sfeat = z.reshape(B, T, P, self.hidden)
        sfeat = sfeat * time_mask[:, :, None, None] * player_mask[:, None, :, None]

        feat = self.fuse(torch.cat([tfeat, sfeat], dim=-1))
        feat = feat * time_mask[:, :, None, None] * player_mask[:, None, :, None]

        # Last valid input frame; with our preprocessing this is always index T-1.
        last = feat[:, -1]
        main = self.motion_head(last).view(B, P, self.horizon, 2).permute(0, 2, 1, 3)
        inter = self.inter_head(feat[:, :-1])
        endpoint = self.endpoint_head(feat)
        return {"main": main, "inter": inter, "endpoint": endpoint}
