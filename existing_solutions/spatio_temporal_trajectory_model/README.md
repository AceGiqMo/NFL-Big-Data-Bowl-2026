# Spatio-Temporal Trajectory Model — NFL Big Data Bowl 2026 (Prediction)

A from-scratch PyTorch re-implementation of a public leaderboard solution for
**NFL Big Data Bowl 2026 — Prediction** competition: predict per-player (x, y)
trajectories for every frame while the ball is in the air, using only pre-pass
tracking data, the targeted receiver and the pass landing location.
Evaluated with the official RMSE metric on a held-out test week.

Repository layout:
```text
spatio_temporal_trajectory_model/
├── data/            # input_2023_w[01-18].csv, output_2023_w[01-18].csv
├── outputs/         # history.json, test_results.json per experiment
├── figures/         # all plots per experiment
├── checkpoints/     # <exp>_best.pt (EMA weights)
├── notebooks/       # Kaggle-notebook for alternative training
├── model.py         # config, data pipeline, architecture, optimizers, metrics
├── train.py         # CLI entry point: train / evaluate / baselines / eda / summarize
├── visualization.py # pure plotting utilities
├── reproduce.sh     # one command per reported number
└── requirements.txt
```

---

## 1. Reference solution

This project reproduces the ideas of the **public 5th-place solution** of the
NFL Big Data Bowl 2026 Prediction competition. The reference solution's key
points, as published by its authors, are:

- **Delta parameterization** — the model predicts per-frame displacements
  (dx, dy) instead of absolute coordinates; absolute trajectories are recovered
  by a cumulative sum anchored at the release-frame position.
- **RoPE over output horizons** — rotary position embeddings encode *which
  future frame* each embedding is intended to predict.
- **Huber loss** — robust to heavy-tailed trajectory errors (cuts, contests).
- **Muon optimizer** — Newton–Schulz orthogonalized momentum for 2-D weights.
- **EMA (decay = 0.995)** — exponential moving average of weights used for
  evaluation and checkpointing.
- A factorized spatio-temporal encoder (SqueezeFormer over time, Transformer
  over players) followed by a horizon-wise recurrent/Transformer decoder.

**Attribution.** All architecture ideas above belong to the original authors;
this repository contains our **independent implementation** of those ideas
(with our own modifications, see Section 3), created for a university course
project. Borrowed concepts and the competition data are explicitly cited:

1. NFL Big Data Bowl 2026 — Prediction, Kaggle competition & dataset.
2. Public 5th-place solution summary (architecture diagram and key points).
3. Muon optimizer — https://github.com/KellerJordan/Muon.
4. RoPE — Su et al., *RoFormer: Enhanced Transformer with Rotary Position
   Embedding*, 2021.
5. SqueezeFormer — Kim et al., 2022.

---

## 2. Architecture

Notation: `B` batch (plays), `P` players, `S` pre-pass frames, `O` output
frames (ball in air), `D = d_model`.

```text
 input features per (player, frame):
 dx, dy, speed, accel, sin/cos(orientation), sin/cos(direction), time-decay
                                                                        (B,P,S,9)
   │
   ▼  temporal-projection (MLP)                                   (B,P,S,D)
   │
   ▼  ×4  SpatioTemporalBlockV1
   │        ├─ temporal: SqueezeFormerBlock over S   (B*P, S, D)
   │        └─ spatial : TransformerEncoderLayer over P            (B*S, P, D)
   │                     + learned distance-matrix attention bias
   │
   │  gather last VALID pre-pass frame (S_i − 1)  →  release-frame snapshot
   ▼  concat[ h_last , static geometry (7 feats) ] → spatio-projection (B,P,D)
   │
   ▼  ×2  TransformerEncoderLayer over players
   │        + learned distance-matrix attention bias
   │
   │  repeat over horizons O  ⊕  RoPE(horizon index)
   │  concat output-phase features (t_sec, o/O, sin, cos) → out_feat_proj
   ▼                                                                 (B,P,O,D)
   │        ┌─ BiLSTM  ×2 (bidirectional, packed) ─┐
   │        ├─ BiGRU   ×2 (bidirectional, packed) ─┼─ concat + skip → mix_proj
   │        └────────────── (identity) ────────────┘        (B,P,O,D)
   ▼  ×2  SpatioTemporalBlockV2
   │        ├─ temporal: TransformerEncoderLayer over O   (B*P, O, D)
   │        └─ spatial : TransformerEncoderLayer over P   (B*O, P, D)
   │                     + learned distance-matrix attention bias
   ▼  head (MLP)                                            (B,P,O,2) = (dx,dy)
   │
   └─ inference: abs_position(o) = last_xy + Σ_{k≤o} delta(k)
```

**Description.**

- **Phase 1 — pre-pass encoding.** Per-frame kinematic features (deltas,
  speed, acceleration, angle sin/cos, recency weight) are projected to `D` and
  processed by 4 factorized spatio-temporal blocks: a SqueezeFormer block
  (FFN–attention–depthwise-conv–FFN) models each player's individual motion
  over time, then a Transformer layer with a *learned distance-matrix attention
  bias* models player–player interactions within each frame. Factorization
  keeps attention cost at O(P·S² + S·P²) instead of O((S·P)²).
- **Phase 2 — release-frame snapshot.** The token of the last valid pre-pass
  frame (gathered per play, since `S` varies inside a batch) is concatenated
  with static geometry (normalized position, distance to ball landing point,
  distance to targeted receiver, side/role flags) and mixed across players by
  two distance-biased Transformer layers. This is the decision-point state.
- **Phase 3 — horizon decoding.** The snapshot is repeated once per output
  frame; each (player, horizon) token receives a **RoPE** rotation by its
  horizon index plus explicit output-phase features (time since release,
  progress o/O, periodic phase), so one shared head serves any `O`. A
  bidirectional LSTM and GRU (packed sequences) refine each player's horizon
  sequence in parallel with a skip connection; two SpatioTemporalBlockV2 layers
  then enforce temporal consistency per player and multi-agent consistency
  within each predicted frame. A small MLP head emits (dx, dy).
- **Training target.** Huber loss on per-frame deltas, masked to existing
  (player, frame) pairs with `player_to_predict = True`. Coordinates are
  normalized so the offense always attacks +x (`play_direction` flip).

---

## 3. Implementation notes

The repository is an independent implementation; during development the
following **modifications and fixes** were introduced relative to a naive
transcription of the reference design:

1. **NaN-safe padded attention (critical).** Padded players/frames produce
   attention rows whose *every* key is masked; `softmax([-inf, …]) = NaN`, and
   `0 · NaN = NaN` then contaminates all real tokens through the value
   vectors. Fixed by (a) giving fully-masked rows uniform scores ("dead-row
   fix") and (b) replacing padded token embeddings with exact zeros via
   `torch.where` (multiplication by a mask cannot remove NaN).
2. **Variable-length pooling.** The release-frame token is obtained with
   `gather` at index `S_i − 1` per play instead of `h[:, :, -1]`, which would
   read padding for shorter plays in a padded batch.
3. **Broadcast fixes.** `deltas_to_absolute` anchors with `anchor[..., None, :]`
   (was an incorrect axis insertion); every output-phase horizon feature is
   explicitly expanded over the player axis (`prog.expand(B, P, O, 1)`).
4. **Optimizer stabilization.** The reference Muon optimizer diverged at epoch
   0 in our setup (NaN loss immediately). Muon is therefore **opt-in**
   (`Config.use_muon = False` by default): all parameters train with AdamW,
   which converged stably. `build_optimizers` returns a *list*, so the training
   loop transparently handles one or two optimizers. Re-enabling Muon
   (`use_muon=True`) reproduces the reference configuration for ablations.
5. **Mixed precision is opt-in** (`Config.amp = False` by default). When
   enabled, the loss is always computed in float32 under `torch.autocast`
   (fp16) with a `GradScaler`; version-compatible helpers support both old and
   new torch AMP APIs.
6. **Custom masked attention.** `nn.MultiheadAttention`'s 4-D `attn_mask`
   support is version-dependent, so attention is implemented manually with
   key-padding masks and a 4-D additive distance bias.
7. **Robust checkpointing.** A fallback checkpoint is saved if the validation
   metric is NaN/inf or no improvement ever occurred, so `evaluate` never fails
   with `FileNotFoundError` after a diverged run; a warning is printed whenever
   the training loss becomes NaN.
8. **Evaluation hygiene.** All models and baselines are scored by the *same*
   harness on the *same* week-based split (train w01–15 / val w16–17 / test
   w18); confidence intervals use a **block bootstrap over plays** (block = 8,
   1000 resamples) to respect temporal dependence within a week.
9. **Reproducibility.** Fixed seeds for Python/NumPy/PyTorch and DataLoader
   generators, deterministic cuDNN, config dumped into every checkpoint and
   `history.json`, one-command reproduction script.
10. **Hardware compatibility.** P100 (Pascal, sm_60) is not supported by modern
    PyTorch binaries (cuDNN RNN kernels missing) and has no Tensor Cores; the
    documented GPU target is T4 (or any sm_75+ device), CPU/macOS MPS also work.

Default hyperparameters: `d_model=128`, `heads=8`, 4×V1 + 2×pre-spatial + 2×V2
blocks, `rnn_hidden=96`, `dropout=0.1`, `batch=32`, `lr_adamw=3e-4`,
`lr_muon=0.02` (if enabled), `weight_decay=0.01`, warmup 500 steps + cosine,
`huber_delta=0.35`, `ema_decay=0.995`, `grad_clip=1.0`, early stopping
patience 6 on val RMSE.

---

## 4. Testing and reproducing

### 4.1 Data

Place the competition CSVs into `data/` (Kaggle API):

```bash
pip install kaggle
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p data
unzip data/nfl-big-data-bowl-2026-prediction.zip "train" -d data   # Linux/macOS
# Windows PowerShell: tar -xf data\nfl-big-data-bowl-2026-prediction.zip -C data "train"
```

Expected files: `data/input_2023_w01.csv … data/output_2023_w18.csv`.

### 4.2 Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, numpy, pandas, matplotlib, tqdm
bash reproduce.sh                        # eda → baselines → 3 seeds → summarize
```

Or step by step:

```bash
python train.py eda                       # dataset statistics → figures/eda.png
python train.py baselines                 # constant-velocity baseline + 95% CI
python train.py train    --seed 42 --exp-name seed_42
python train.py evaluate --exp-name seed_42
python train.py summarize --exp-names seed_42,seed_43,seed_44
```

### 4.3 Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# reproduce.sh needs Git Bash / WSL; otherwise run the commands of §4.2 manually:
python train.py eda
python train.py baselines
python train.py train    --seed 42 --exp-name seed_42
python train.py evaluate --exp-name seed_42
```

For CUDA wheels on Windows/Linux:

`pip install torch --index-url https://download.pytorch.org/whl/cu121`.

CPU-only machines work but are slow (expect several hours per seed); for GPU
runs without local hardware use Kaggle (§4.4). **macOS MPS** is auto-detected but
also slow for this model.

### 4.4 Kaggle notebook (recommended for GPU training)

1. Upload the notebook from `./notebooks` folder, **Add input** → the competition dataset.
2. Settings → Accelerator → **GPU T4 x2** (do **not** use P100: Pascal/sm_60
   kernels are absent from modern torch binaries and fp16 has no Tensor Cores).
3. Upload this folder (without `data/`) as a Kaggle dataset, then in a cell
4. Download `artifacts.zip` from the Output pane and unpack it into the local
   repository (`checkpoints/`, `outputs/`, `figures/` merge automatically).
5. Verify reproducibility locally: `python train.py evaluate --exp-name seed_42`
   must reproduce the Kaggle RMSE within ~1e-3 (evaluation is deterministic).

Typical runtimes: data preparation 3–6 min (once), training 15–25 min per seed
on T4 (fp32), evaluation 2–3 min.

### 4.5 Sanity checks

```python
# quick shape/NaN self-test before spending GPU time
from model import TrajectoryModel, Config, make_loader, load_all_plays, huber_delta_loss
import torch
cfg = Config()
batch = next(iter(make_loader(
    __import__("model").NFLDataset(load_all_plays(cfg), cfg, cfg.train_weeks, {}),
    cfg, False)))
out = TrajectoryModel(cfg)(batch)
loss = huber_delta_loss(out.float(), batch["dY"],
                        batch["valid"] * batch["to_pred"][..., None], cfg.huber_delta)
loss.backward()
assert out.shape[-1] == 2 and torch.isfinite(out).all() and torch.isfinite(loss).all()
```

---

## 5. Results

Test set: week 18 (750 plays). All numbers from the same evaluation harness;
95% CI by block bootstrap over plays (1000 resamples). EMA weights, seed 42.

| Model | Test RMSE (yards) | 95% CI |
|---|---|---|
| **Proposed (EMA)** | **0.80** | [0.70, 0.95] |
| Constant-velocity baseline | 4.25 | [3.96, 4.52] |

The proposed model is **≈5.3× more accurate** than the naive baseline; the
confidence intervals do not overlap, so the improvement is statistically
significant. Validation: best val RMSE ≈ **1.02** at epoch 18 (early stopping at
epoch 24, patience 6); train/val Huber curves show healthy convergence with
only mild overfitting after epoch ~13.

**Error profile** (see `figures/seed_42/`):

- Median radial error ≈ **0.28 yd**, p90 ≈ 1.6 yd, p99 ≈ 3.6 yd — heavy-tailed
  distribution (sharp cuts, contested catches), which motivates the Huber loss
  and explains RMSE ≫ median (`error_qq.png`).
- Error growth over horizon: RMSE(o) rises from ≈0.2 yd (o≈2) to ≈3.7 yd
  (o=30) due to cumsum drift of per-frame deltas, versus 1.0 → 15.4 yd for the
  baseline; the model's advantage grows with pass length (`rmse_vs_horizon.png`).
- Per-role stratification: Targeted Receiver median per-player RMSE ≈ 0.19 yd
  (IQR 0.11–0.35) vs Defensive Coverage ≈ 0.30 yd (IQR 0.16–0.58) — reactive
  defenders are the hardest stratum (`role_breakdown.png`).
- Qualitative trajectory examples and the model-vs-baseline comparison:
  `trajectories.png`, `rmse_comparison.png`; training dynamics:
  `training_history.png`.

**Multi-seed stability:** seeds 43 and 44 are scheduled; the aggregated
mean ± std will be reported in `outputs/seed_summary.json` and
`figures/seed_summary.png` after `python train.py summarize
--exp-names seed_42,seed_43,seed_44` (placeholder: *TBD*).

**Known limitations / next steps** (candidates for the "model improvements"
stage): cumsum drift on long horizons (auxiliary absolute-position loss or
velocity-residual parameterization), heavy error tail (horizon- or
role-weighted loss), and re-enabling Muon/AMP as controlled ablations once
stability guards are in place.