# HAST-Net: Horizon-Anchored Spatio-Temporal Network for NFL Trajectory Forecasting

Course project for **NFL Big Data Bowl 2026 — Prediction**: predict per-player
`(x, y)` trajectories for every frame while the ball is in the air, using only
pre-pass tracking data, the targeted receiver and the pass landing location.
Evaluated with the official RMSE metric on a held-out test week (week 18).

```text
hastnet/
├── model.py           # config, data pipeline, HAST-Net architecture, losses, optimizers, metrics
├── train.py           # CLI entry point: --mode train | tune | smoke_test
├── visualization.py   # pure plotting utilities
├── outputs/           # history.json, config.json, test_results.json
├── figures/           # all plots per experiment
├── checkpoints/       # <exp>_best.pt (EMA weights)
├── tuning/            # tuning checkpoints, figures, tuning_results.json
├── hastnet_artifacts.zip          # baseline run artifacts
└── hastnet_tuning_artifacts.zip   # tuning run artifacts
```

## 1. Reference solution

HAST-Net is our own architecture, but it reuses proven ideas from the public
top solutions of the competition that were reproduced earlier in this project:

- **5th place** — delta parameterization with cumsum recovery anchored at the
  release frame; factorized spatio-temporal encoder (SqueezeFormer over time,
  Transformer over players); Huber loss; EMA weights.
- **3rd place** — auxiliary targets (next-frame displacement, endpoint)
  alongside the main future-position loss.
- **1st place** — probabilistic prediction (Gaussian NLL), auxiliary
  velocity/acceleration objectives, spatial augmentations (rotation, flip,
  x/y shift) and temporal frame shift.

Novel components of HAST-Net relative to these references: a **horizon-anchor
head** (per-player sigmoid schedule between the release position and a learned
endpoint) and a **coarse-to-fine control-point decoder** (Fourier queries →
cross-attention → BiGRU → linear interpolation), which remove the cumsum drift
of pure delta parameterization at medium and long horizons.

Attribution: the competition data and all borrowed concepts are cited in
Section 6; the reproduction reports are separate deliverables of this project.

## 2. Architecture

Notation: `B` batch (plays), `P` players, `S` pre-pass frames, `O` output
frames (ball in air), `D = d_model`, `M` control points.

```text
 input features per (player, frame):
  dx, dy, speed, accel, sin/cos(dir), sin/cos(orient), recency        (B,P,S,9)
   │  stem (MLP)                                                     (B,P,S,D)
   ▼  ×4 SpatioTemporalBlockV1
   │     ├─ temporal: SqueezeFormerBlock over S        (B*P, S, D)
   │     └─ spatial : TransformerEncoderLayer over P   (B*S, P, D)
   │                  + learned distance-matrix attention bias
   │  gather last VALID pre-pass frame (S_i − 1) → release snapshot
   ▼  concat[ h_last, static geometry (7 feats) ] → MLP             (B,P,D)
   ▼  ×2 TransformerEncoderLayer over players + distance bias
   │  global play token: learned query × cross-attn over players per frame
   ▼
   horizon-anchor head (per player):
     τ, w heads;  anchor(o) = last_xy + (end − last_xy)·σ((o/O − τ)/w)  (B,P,O,2)
   ▼
   coarse-to-fine control-point decoder:
     M Fourier queries ⊕ player state → cross-attn to [players ⊕ global token]
     → BiGRU → M residual control points → linear interpolation over o
   ▼
   refinement MLP (+ horizon Fourier features) → ρ(o), log σ²(o)
   ▼
   pos(o) = anchor(o) + ρ(o);   Gaussian NLL on (pos, σ²)
```

### Design decisions and training objective

- **Factorized attention** keeps attention cost at O(P·S² + S·P²) instead of
  O((S·P)²).
- **Anchor head** makes the pass endpoint an explicit attractor with a
  per-player learned schedule (τ, w), so medium/long horizons do not drift the
  way a pure cumsum of deltas does.
- **Control-point decoder** emits M = 8 coarse residuals refined by a BiGRU
  and interpolated continuously over the horizon, so one shared head serves
  any `O`.
- **Probabilistic head** (Gaussian NLL) provides calibrated per-frame
  uncertainty.

Loss (masked to valid (player, frame) pairs with `player_to_predict = True`):

L = w_pos·Huber(pos) + w_end·Huber(end) + w_dec·Huber(Δ, exp-decay) + w_acc·Huber(accel) + w_nll·NLL

with weights (1.0, 0.5, 0.5, 0.1, 0.1) and `huber_delta = 0.35`. Coordinates are
normalized so the offense always attacks +x (`play_direction` flip).

## 3. Implementation notes

- **NaN-safe padded attention (critical).** Fully-masked attention rows give
  `softmax([-inf, …]) = NaN`, and `0 · NaN = NaN` then contaminates all real
  tokens through the value vectors. Fixed by (a) uniform scores for
  fully-masked rows ("dead-row fix") and (b) replacing padded token embeddings
  with exact zeros via `torch.where`.
- **Variable-length pooling.** The release-frame token is obtained with
  `gather` at index `S_i − 1` per play instead of `h[:, :, -1]`, which would
  read padding for shorter plays in a padded batch.
- **Broadcast fixes.** Anchor broadcasting uses explicit
  `last[:, :, None, :]` / `end[:, :, None, :]` insertions; every horizon
  feature is explicitly expanded over the player axis.
- **Optimizer stabilization.** Muon diverged at epoch 0 in our setup, so it is
  opt-in (`use_muon = False` by default); all parameters train with AdamW +
  warmup + cosine. Mixed precision is opt-in (`amp = False`); when enabled, the
  loss is always computed in float32 under autocast with a GradScaler.
- **Robust checkpointing.** A fallback checkpoint is saved if the validation
  metric is NaN/inf or no improvement ever occurred, so evaluation never fails
  after a diverged run.
- **Evaluation hygiene.** All models and baselines are scored by the same
  harness on the same week-based split (train w01–15 / val w16–17 / test w18);
  95% CI by block bootstrap over plays (block = 8, 1000 resamples).
- **Reproducibility.** Fixed seeds (Python/NumPy/PyTorch/DataLoader),
  deterministic cuDNN, config dumped into every checkpoint and `history.json`.
- **Hardware.** Target GPU is T4 (sm_75+); P100/Pascal is not supported by
  modern torch binaries (missing cuDNN RNN kernels, no Tensor Cores). CPU and
  macOS MPS work but are slow.

```pycon
@dataclass
class Config:
    d_model: int = 128
    n_heads: int = 8
    n_v1_blocks: int = 4
    n_pre_spatial: int = 2
    n_control: int = 8          # M control points of the coarse-to-fine decoder
    rnn_hidden: int = 96
    dropout: float = 0.1
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500     # + cosine schedule
    huber_delta: float = 0.35
    ema_decay: float = 0.995
    grad_clip: float = 1.0
    patience: int = 6           # early stopping on val RMSE
    epochs: int = 25
    seed: int = 42
```

## 4. Testing and reproducing

### 4.1 Data

Place the competition CSVs into `data/` (Kaggle API). Expected files:
`data/input_2023_w01.csv … data/output_2023_w18.csv`.

```bash
pip install kaggle
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p data
unzip data/nfl-big-data-bowl-2026-prediction.zip -d data
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy, pandas, matplotlib, tqdm

# quick pipeline self-test: 10 plays from week 01, 1 epoch
python train.py --mode smoke_test

# full training + evaluation + figures + artifacts zip
python train.py --mode train --seed 42

# hyperparameter tuning (RandomizedSearch) + tuning artifacts zip
python train.py --mode tune --n-trials 3
```

```py
# shape/NaN self-test before spending GPU time
import torch
from model import HASTNet, Config, HASTDataset, collate, hast_loss

cfg = Config()
ds = HASTDataset(idx_df, cfg, cfg.train_weeks, PREP, train=True)
batch = collate([ds[i] for i in range(4)])
out = HASTNet(cfg)(batch)
loss, parts = hast_loss(out, batch, cfg)
loss.backward()
assert out["pos"].shape == batch["pos"].shape
assert torch.isfinite(out["pos"]).all() and torch.isfinite(loss).all()
```

## 5. Results

Test set: week 18 (750 plays). All numbers from the same evaluation harness;
95% CI by block bootstrap over plays (1000 resamples). EMA weights, seed 42.

| Model | Test RMSE (yards) | 95% CI |
| --- | --- | --- |
| HAST-Net (EMA) | 0.838 | [0.762, 0.913] |
| Constant-velocity baseline | 1.702 | [1.589, 1.812] |

The model is ≈2.0× more accurate than the naive baseline; the confidence
intervals do not overlap, so the improvement is statistically significant.
Validation: best val RMSE 0.948 at epoch 23 (25-epoch schedule, patience 6);
train/val curves show healthy convergence with only mild overfitting after
epoch ~18.

Cross-project reference points (different protocols, not directly comparable):
5th-place reproduction test 0.80 [0.70, 0.95] on the same split; 3rd-place
reproduction val 1.365 (weeks 1–5); 1st-place reproduction val CV-RMSE 0.4538
(GroupKFold, 48-frame horizon).

### Error profile

- Median radial error ≈ 0.2 yd, p90 ≈ 1.4 yd, p95 ≈ 2.0 yd, p99 ≈ 3.3 yd —
  heavy-tailed distribution (sharp cuts, contested catches), which motivates
  the Huber loss and explains RMSE ≫ median.
- Error growth over horizon: RMSE(o) grows from ≈0.01 yd (o = 1) to the
  multi-yard level at o ≥ 25; the longest bins contain only a handful of
  plays, so their CIs are wide. An earlier double-squaring bug in the
  per-horizon metric was found and fixed.
- σ-calibration is close to ideal for σ ≤ 0.8, with mild mis-calibration in
  the mid/top bins — usable for uncertainty-aware gating.

### Hyperparameter tuning (RandomizedSearch)

Three configurations, 12 epochs each, selection by best EMA val RMSE; the
winner is re-evaluated on the test week:

| Config | Changes | Params | Best val RMSE |
| --- | --- | --- | --- |
| baseline | — | 2.97M | 0.948 |
| tune_capacity | d_model 192, n_control 12, rnn_hidden 128 | 6.52M | 1.021 |
| tune_regaug | dropout 0.2, wd 0.05, stronger rot/shift/frame-shift aug | 2.97M | 1.084 |
| tune_loss_tail | w_end 1.0, w_dec 0.7, w_nll 0.3, huber 0.5, lr 4e-4 | 2.97M | 1.014 |

Winner `tune_loss_tail`: test RMSE 0.891 [0.811, 0.975] — statistically
indistinguishable from the baseline but worse in point estimate, so the
baseline configuration remains the final model. Negative result: at this
data/compute budget the default configuration is near-optimal; extra capacity
is undertrained at 12 epochs, and heavy augmentation hurts.

## 6. Findings and limitations

**Findings.**

- The horizon-anchor head plus control-point decoder removes cumsum drift at
  medium horizons and yields interpretable per-player endpoint schedules.
- The global play token and distance-biased spatial attention capture
  multi-agent context; reactive defenders (Defensive Coverage) remain the
  hardest stratum.
- The probabilistic head calibrates well and provides free per-frame
  uncertainty.
- RandomizedSearch over three configurations did not beat the default —
  reported as a valid negative result.

**Limitations / next steps.**

- Long-horizon tail (o ≥ 25): a handful of broken plays dominates the squared
  error; endpoint-gated blending with a constant-velocity fallback at high σ
  is the main improvement candidate.
- Single seed (42) for the headline number; multi-seed aggregation pending.
- No pretraining, ensembling or test-time augmentation (out of scope).
- The per-horizon metric double-squaring bug was fixed post hoc; earlier
  printed horizon numbers were inflated.

## References

- NFL Big Data Bowl 2026 — Prediction, Kaggle competition & dataset.
- Public 5th-place solution summary (delta parameterization, factorized
  spatio-temporal encoder, EMA); Muon — KellerJordan; RoPE — Su et al., 2021;
  SqueezeFormer — Kim et al., 2022.
- Public 3rd-place write-up: "PreTrain-FineTune: Spatio-Temporal Transformer
  with Multi-Aux-Loss for NFL Trajectory Forecasting."
- Public 1st-place notebook reproduction (Gaussian NLL, auxiliary
  velocity/acceleration heads, EMA, GroupKFold).
- Project deliverables: EDA notebook (`EDA.ipynb`, `EDA_plots.pdf`) and the
  reproduction reports (`5th_place_simulation_report.md`,
  `3rd_place_simulation_report.md`, `1st_place_simulation_report.md`).
