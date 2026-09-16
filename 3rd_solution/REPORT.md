# NFL Big Data Bowl 2026 — 3rd Place Solution Reproduction

## 1. Problem statement

The competition task is to predict where each offensive/defensive player
will move after the ball is thrown, given tracking data (position, speed,
acceleration, direction, orientation, role) recorded before the throw at
10 Hz. The target is (x, y) for every "player to predict," for every frame
until the ball is caught or hits the ground — a variable-length horizon
that differs per play and per player.

## 2. Reference solution

The reproduced approach follows the public 3rd-place write-up,
*"PreTrain-FineTune: Spatio-Temporal Transformer with Multi-Aux-Loss for NFL
Trajectory Forecasting."* Its three central ideas:

- **Spatio-Temporal Transformer** — attention along two axes: temporal
  (how a player's own state evolves frame to frame) and spatial (how
  players relate to each other within a single frame).
- **Multi-Aux-Loss** — auxiliary training targets (next-frame displacement,
  endpoint) alongside the main future-position loss, to encourage a richer
  internal representation.
- **PreTrain-FineTune** — broad pretraining followed by fine-tuning on the
  competition objective.

This repository implements the documented core architecture and multi-task
loss design. It does not include 2018-season pretraining, augmentation, EMA,
test-time augmentation, k-fold cross-validation, or ensembling, which are out
of scope for this project.

## 3. Architecture

```
Input  X: [B, T, P, F]   (batch, time steps, up to 22 player slots, features)
           │
           ▼
   input_proj (Linear → GELU → Linear)         per (time, player) frame
           │
           + learned time embedding[:T]
           + learned player-slot embedding[:P]
           │  (zeroed for padding slots via player_mask)
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
   last valid observed frame (index T-1 by construction)
           │
  ┌────────┼─────────────────────┐
  ▼        ▼                     ▼
 main    inter (aux)         endpoint (aux)
 (B,H,P,2)  next-frame Δ      displacement to
 future    inside observed    the final future
 Δx, Δy    history            point, from every
                               observed frame
```

**Design decisions:**

- *22 fixed player slots*, offense sorted before defense, both ordered by
  distance to the passer at the first observed frame, so each slot carries a
  consistent role across plays.
- *Two sequential attention stacks* (temporal, then spatial) rather than a
  single fused spatio-temporal block, favoring simplicity and debuggability
  over maximal expressiveness.
- *Displacement targets* (Δx, Δy from the last observed point) rather than
  absolute coordinates, keeping the regression scale small and centered.
- *Time-decayed Huber loss* (`exp(-0.03·t)` weighting) as the main loss,
  plus small velocity/acceleration smoothness penalties and two auxiliary
  losses (next-frame displacement, running endpoint estimate).
- *Left-padding with explicit masks* for variable-length observation windows
  and missing players; attention key-padding masks exclude padding from every
  computation.

Relative to the published solution, this implementation omits: a separate
player-interaction aggregation sub-path, an explicit sparse positional
encoding module (using additive learned embeddings instead), and a third
auxiliary head for full-temporal reconstruction.

## 4. Implementation notes

The following correctness considerations were identified and addressed
during development:

- **Endpoint supervision.** Horizons vary per play and per player. Endpoint
  targets use each target player's own last valid future frame (via a
  `gather` on the per-player valid-frame count), not the batch-padded final
  frame — otherwise shorter trajectories would be supervised against an
  artificial endpoint.
- **Validation RMSE aggregation.** Computed globally as total squared error
  divided by the total number of valid (x, y) coordinates across the whole
  validation set, rather than averaging per-batch RMSE values — the latter
  is a biased estimator once batches contain a different number of valid
  coordinates.
- **Variable-horizon batching.** The model always emits the dataset-wide
  maximum horizon, while a given batch's targets are only padded to that
  batch's own (possibly smaller) maximum horizon. Model outputs are sliced
  to the batch horizon before the loss/metric is computed. Without this,
  batches that omit the longest play in the dataset raise a tensor
  size-mismatch error.
- **Trajectory visualization.** `plot_trajectory.py` filters observed history
  with `time_mask` before plotting, and reconstructs the model with the
  checkpoint's stored `max_time`, so it renders correctly for any
  `--window` value.

## 5. Testing

Real Kaggle CSVs are large (~865 MB per week) and outside this repository's
scope to redistribute. To validate correctness independent of the data
volume, the pipeline (`eda.py` → `preprocess.py` → `baseline.py` →
`train.py` → `plot_training.py` → `plot_trajectory.py`) was exercised
end-to-end on synthetic tracking data constructed to stress the two edge
cases padding relies on:

- **Variable observation windows** (8–14 frames, left-padded to a fixed
  window): confirmed `time_mask` is exactly 0 on padded frames and 1 on real
  ones, and that the plotted history starts at the true first observed
  frame.
- **Variable output horizons** (12 vs. 28 frames across plays in the same
  batch): confirmed training completes without shape errors and validation
  RMSE aggregates correctly across batches of different horizons.

This confirms the pipeline is functionally correct end-to-end; it is not a
substitute for training on the full dataset.

## 6. Reproducing results

The results in Section 7 were produced with:

```bash
python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda
python src/preprocess.py --data-dir data/train --weeks 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 \
    --window 20 --output data/processed/weeks-all.pkl
python src/baseline.py --data data/processed/weeks-all.pkl
python src/train.py --data data/processed/weeks-all.pkl --epochs 10 \
    --batch-size 8 --hidden 192 --layers 2 --output checkpoints/model.pt
python src/plot_training.py --history outputs/training_history.csv \
    --output outputs/training_curves.png
python src/plot_trajectory.py --data data/processed/weeks-all.pkl \
    --checkpoint checkpoints/model.pt --index 0 --player-slot 0 \
    --output outputs/trajectory.png
```

## 7. Results

| Model | Validation RMSE (yards) |
|---|---|
| Stationary baseline | 4.351 |
| Constant-velocity baseline | 5.958 |
| STTransformer (this repository) | 0.831 |

Trained on all 18 weeks of the 2023 season (2,765 plays held out for
validation), 10 epochs, hidden size 192, 2 layers. The model outperforms the
stationary baseline by ~81% and the constant-velocity baseline by ~86%.
Validation RMSE improved consistently as more data was added: 4.020 on a
single week, 1.365 on five weeks, 0.831 on the full 18 weeks — the model
keeps extracting more signal from player-specific and route-specific
movement patterns as training data grows, while the fixed-rule baselines
stay flat. The constant-velocity baseline remains worse than the stationary
one throughout, for the reason noted earlier: players frequently change
speed and direction while the ball is in the air.

Training/validation curves: `outputs/training_curves.png`. Example
predicted vs. actual trajectory: `outputs/trajectory.png`.
