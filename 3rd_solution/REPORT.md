# NFL Big Data Bowl 2026 — 3rd Place Solution Reproduction

## 1. Task

The competition predicts where each offensive/defensive player will move **after
the ball is thrown**, using tracking data (x, y, speed, acceleration, direction,
orientation, roles) recorded **before** the throw. Input: a sequence of frames
(10 Hz) up to the throw. Output: (x, y) for every "player to predict" for every
frame until the ball is caught or hits the ground (a variable-length horizon
per play).

## 2. What the public 3rd-place write-up describes (high level)

Title: *"PreTrain-FineTune: Spatio-Temporal Transformer with Multi-Aux-Loss for
NFL Trajectory Forecasting."* The three ideas in that name:

- **Spatio-Temporal Transformer** — attention along two axes: *temporal*
  (how one player's own state evolves frame to frame) and *spatial*
  (how players relate to each other within a single frame).
- **Multi-Aux-Loss** — besides the main "future (x, y)" loss, the model is
  also trained on auxiliary targets (e.g. next-frame displacement, final
  endpoint) so it learns a better internal representation, not just the
  end coordinates.
- **PreTrain-FineTune** — train broadly first (more data / an easier
  objective), then fine-tune on the exact competition target.

**Scope note:** The original Kaggle write-up is public and explicitly describes the
feature set, 22-player slot construction, dual-path architecture, four auxiliary
objectives, and the full training pipeline. This repository intentionally
implements a smaller educational subset of those ideas rather than claiming
byte-for-byte or leaderboard parity.

## 3. What this repository actually implements

This is a deliberately **minimal, transparent educational reproduction** —
not a leaderboard clone — of the core idea, built to be understandable and
trainable inside a course timeline.

```
Input  X: [B, T, P, F]   (batch, time steps, up to 22 player slots, features)
           │
           ▼
   input_proj (Linear → GELU → Linear)         per (time, player) frame
           │
           + learned time embedding[:T]
           + learned player-slot embedding[:P]
           │  (zeroed out for padding slots via player_mask)
           ▼
 ┌─────────────────────────────┐
 │ Temporal attention ×layers   │  each player attends over its own T frames
 │ (key_padding_mask = ~time_mask)
 └─────────────────────────────┘
           │
 ┌─────────────────────────────┐
 │ Spatial attention ×layers    │  at each frame, players attend to each other
 │ (key_padding_mask = ~player_mask)
 └─────────────────────────────┘
           │
   concat(temporal_feat, spatial_feat) → fuse (LayerNorm → Linear → GELU)
           │
           ▼
   take the LAST valid observed frame (index T-1 by construction)
           │
  ┌────────┼─────────────────────┐
  ▼        ▼                     ▼
 main    inter (aux)         endpoint (aux)
 (B,H,P,2)  next-frame Δ      displacement to
 future    inside observed    the final future
 Δx, Δy    history            point, from every
                               observed frame
```

Design choices and why:
- **22 fixed player slots**, offense sorted before defense, both ordered by
  distance to the passer at the first observed frame — gives the model a
  consistent "meaning" per slot across plays instead of an arbitrary order.
- **Two separate attention stacks** (temporal-then-spatial) rather than one
  fused spatio-temporal block — simpler to implement and debug, at the cost
  of being less expressive than a truly joint block.
- **Displacement target** (Δx, Δy from the last observed point), not absolute
  coordinates — keeps the regression scale small and centered near zero.
- **Time-decayed Huber loss** as the main loss (`exp(-0.03·t)` weight) plus
  small velocity/acceleration smoothness penalties, plus two auxiliary
  losses (next-frame displacement, running endpoint estimate).
- **Left-padding + masks** for variable-length observed windows and missing
  players — attention key-padding masks make sure padding never contributes.

### Compared with the original 3rd-place solution, this version omits:
- the separate "player interaction" aggregation sub-path (summarize all
  players into one context vector per frame before the main path);
- an explicit sparse positional encoding module (uses simple additive
  learned embeddings instead);
- a third auxiliary head ("full temporal prediction" from every observed
  frame, not just the last one);
- 2018-season pretraining + fine-tuning, augmentation, EMA, test-time
  augmentation, k-fold CV and ensembling — all explicitly listed as later
  add-ons in `README_FIRST.md`, once the core model is verified to work.

None of these omissions are bugs — they're a scope decision documented in the
README so it's defensible in a presentation: *"we reproduced the documented
core architecture and multi-task loss idea; we did not attempt full leaderboard
parity given the course timeline."*

## 4. Bugs found and fixed

`src/plot_trajectory.py` plotted the **raw, un-masked** `X` history, including
the zero-padded frames added by `preprocess.py` for short plays. This was fixed
by filtering the history with `time_mask`.

Two additional correctness fixes are included in the final version:

- `src/losses.py`: endpoint supervision now uses the **last valid future frame
  for each player**. The competition explicitly allows a different
  `num_frames_output` / horizon for each `game_id`/`play_id`/`nfl_id`, so using
  the batch-padded last frame would create an artificial endpoint for shorter
  trajectories. citeturn0search0
- `src/train.py`: validation RMSE is now computed globally from the total
  squared error divided by the total number of valid x/y coordinates. This is
  the correct aggregation for the competition metric when plays have
  variable-length outputs. citeturn0search3

- `src/train.py`: this change also fixes a **shape-mismatch crash**. The
  model always outputs the dataset-wide max horizon, but a batch's targets
  are only padded to that batch's own (possibly smaller) max horizon; outputs
  are now sliced to match before the loss/metric. Confirmed with a synthetic
  10-play set (9 plays horizon 12, 1 play horizon 28): the previous version
  crashed with a tensor size mismatch on batches without the horizon-28 play;
  the current version trains cleanly.
- `src/plot_trajectory.py`: `load_model` now passes `max_time` through when
  rebuilding the model, so the script works with any `--window` value used
  during preprocessing, not only the default of 20.

The project was re-run on synthetic data after these changes.

## 5. Pipeline validation (synthetic data, this session)

Real Kaggle CSVs are not available in this environment, so the full pipeline
(`eda.py` → `preprocess.py` → `train.py` → `plot_trajectory.py`) was run on a
hand-built synthetic dataset (12 plays, 2 games, variable-length windows of
8–14 observed frames to exercise the left-padding path) to check for runtime
bugs, not to measure real accuracy:

- `preprocess.py`: 12/12 plays converted successfully; masks and displacement
  targets were spot-checked and match expectations (verified `time_mask` is
  `0` exactly on the padded frames and `1` on real ones).
- `train.py`: 3 epochs, training loss decreased monotonically each epoch,
  validation split by `game_id` worked (no leakage), checkpoint saved.
- `plot_trajectory.py`: after the fix, the plotted history segment starts at
  the true first observed frame, not at the origin.

This confirms the pipeline is **functionally correct end-to-end**; it does not
and cannot substitute for training on real weeks of data.

## 6. How to run on the real data

```bash
# 1) EDA on one week
python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda

# 2) Build a small dev set first
python src/preprocess.py --data-dir data/train --weeks 1 --window 20 \
    --max-plays 500 --output data/processed/dev.pkl

# 3) Smoke-test training
python src/train.py --data data/processed/dev.pkl --epochs 3 --batch-size 8 \
    --hidden 128 --layers 1 --output checkpoints/smoke.pt

# 4) Scale up once the smoke test looks sane
python src/preprocess.py --data-dir data/train --weeks 1 2 3 --window 20 \
    --max-plays 5000 --output data/processed/train_small.pkl
python src/train.py --data data/processed/train_small.pkl --epochs 10 \
    --batch-size 8 --hidden 192 --layers 2 --output checkpoints/model.pt

# 5) Visualize a prediction
python src/plot_trajectory.py --data data/processed/train_small.pkl \
    --checkpoint checkpoints/model.pt --index 0 --player-slot 0
```

Fill in `outputs/eda/summary.csv` and the generated plots from step 1, and the
per-epoch `train_loss` / `val_RMSE` printout from step 3–4, directly into the
"Results" section below once you have access to the real data.

## 7. Results (fill in with real numbers)

| epoch | train loss | val RMSE (yards) |
|---|---|---|
| … | … | … |

Add: a trajectory plot for 2–3 example plays, and a small table comparing the
neural model with the stationary and constant-velocity baselines from
`src/baseline.py`. This is the most convincing sanity check that the model
learned something beyond a trivial extrapolation.

## 8. What to say in the presentation / report

1. The task is next-frame trajectory forecasting conditioned on tracking data.
2. Players are placed in 22 fixed slots so the model sees a consistent input
   layout across plays; masks handle missing players and short histories.
3. Temporal attention models how a player's own motion evolves; spatial
   attention models interaction between players at a given moment.
4. The model predicts future Δx/Δy from the last observed point; auxiliary
   losses (next-frame delta, running endpoint) encourage temporally
   consistent, endpoint-aware representations.
5. This is a transparent educational reproduction of the write-up's
   documented core ideas, not a byte-for-byte leaderboard clone — the report
   explicitly lists what was intentionally left out and why (see §3).
