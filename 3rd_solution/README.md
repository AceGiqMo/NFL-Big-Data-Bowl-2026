# NFL Big Data Bowl 2026 — 3rd Place Solution: minimal educational reproduction

This is a **minimal reproduction for a university Practical ML/DL project**, not a leaderboard clone.

It implements the core ideas stated in the public 3rd-place Kaggle write-up:
- published feature engineering;
- 22 fixed player slots, offense/defense ordered by distance to passer at the first frame;
- dual-path spatio-temporal attention (temporal per-player + spatial inter-player);
- player/time embeddings and masks;
- future displacement (Δx, Δy) prediction;
- primary temporal-weighted Huber-style loss plus inter-frame and endpoint auxiliary losses;
- AdamW, cosine LR schedule and gradient clipping.

The original competition solution additionally used 2018 pretraining, fine-tuning, augmentations, EMA, TTA, 7-fold CV and ensembling. Those are intentionally left out of the minimal version so the architecture can be understood and trained in a course project.

## 0. Folder layout

```text
data/train/input_2023_w01.csv
...
data/train/output_2023_w01.csv

src/features.py
src/preprocess.py
src/model.py
src/losses.py
src/train.py
src/plot_trajectory.py
notebooks/eda.py
```

## 1. Download data

Download the train folder from Kaggle and put the CSVs under `data/train/`.

## 2. EDA

```bash
python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda
```

## 3. Build a small training set

Start small:

```bash
python src/preprocess.py --data-dir data/train --weeks 1 --window 20 --max-plays 1000 --output data/processed/week1_1000.pkl
```

## 4. Compare simple baselines

```bash
python src/baseline.py --data data/processed/week1_1000.pkl
```

The validation split is grouped by `game_id`. The stationary baseline predicts zero future displacement; the constant-velocity baseline extrapolates the final observed velocity at 10 Hz.

## 5. Smoke-test training

```bash
python src/train.py --data data/processed/week1_1000.pkl --epochs 3 --batch-size 8 --hidden 128 --layers 1 --output checkpoints/smoke.pt
```

Then use a larger configuration:

```bash
python src/preprocess.py --data-dir data/train --weeks 1 2 3 --window 20 --max-plays 5000 --output data/processed/train_small.pkl
python src/train.py --data data/processed/train_small.pkl --epochs 10 --batch-size 8 --hidden 192 --layers 2 --output checkpoints/third_place_minimal.pt
```

## 6. Visualize a prediction

```bash
python src/plot_trajectory.py --data data/processed/train_small.pkl --checkpoint checkpoints/third_place_minimal.pt --index 0 --player-slot 0
```

## What to say in the report

1. Input is a sequence of tracking frames for one play.
2. Up to 22 players are represented in fixed slots.
3. Temporal attention models how each player's motion evolves.
4. Spatial attention models interactions between players at the same time.
5. The model predicts future Δx/Δy; adding them to the last observed coordinates gives future x/y.
6. Auxiliary losses encourage temporal consistency and endpoint awareness.

Do not claim this is an exact reproduction of the Kaggle leaderboard model. It is a transparent educational reproduction of its core architecture.

## Important implementation notes

This reproduction intentionally keeps the model educational. The public 3rd-place write-up does not expose every internal implementation detail, so this project reproduces the documented architectural ideas rather than claiming byte-for-byte parity.

### Fixes compared with the first draft

- Validation is split by `game_id`, avoiding train/validation leakage between plays from the same game.
- Input sequences shorter than the fixed window are explicitly left-padded and have a `time_mask`.
- Main loss uses only `player_to_predict` targets.
- Auxiliary inter-frame and endpoint losses respect both time and player masks.
- Endpoint supervision uses each target player's own last valid future frame, rather than the batch-padded final frame.
- Validation RMSE is aggregated globally from total squared error / total valid coordinates, so variable-length horizons are weighted correctly.
- `src/baseline.py` provides stationary and constant-velocity baselines for a meaningful comparison.
- The main loss includes the documented exponential time decay and small velocity/acceleration smoothness terms.
- The target is future displacement from the last observed coordinates, matching the solution's described formulation.
- A Jupyter EDA notebook plus a compact EDA script were added.

### What the EDA should answer

The EDA is deliberately small and model-oriented. It checks shapes/missingness, number of players and frames per play, prediction horizon, roles/positions, motion distributions, an example multi-player trajectory, and receiver-to-ball-landing distance. These findings justify sequence modeling, fixed player slots, masks, and temporal/spatial attention.
