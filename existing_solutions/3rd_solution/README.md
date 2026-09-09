# NFL Big Data Bowl 2026 — 3rd Place Solution Reproduction

Reproduction of the core architecture and training approach described in the
public 3rd-place write-up of the *NFL Big Data Bowl 2026 — Prediction*
Kaggle competition ("PreTrain-FineTune: Spatio-Temporal Transformer with
Multi-Aux-Loss for NFL Trajectory Forecasting"). Given pre-throw player
tracking data, the model predicts each targeted player's (x, y) trajectory
after the ball is thrown.

See `REPORT.md` for the full methodology, validation, and results.

## Approach

- 22 fixed player slots per play (offense before defense, both ordered by
  distance to the passer at the first observed frame), so the model sees a
  consistent input layout across plays.
- A dual-path attention encoder: temporal attention lets each player attend
  over its own observed frames; spatial attention lets players at the same
  frame attend to each other.
- The model predicts future displacement (Δx, Δy) from the last observed
  position, supervised with a time-decayed Huber loss plus velocity/
  acceleration smoothness terms.
- Two auxiliary heads — next-frame displacement and running endpoint
  estimate — provide additional training signal.
- Left-padding and explicit time/player masks handle variable-length
  observation windows and missing players.

## Scope

This repository implements the documented core architecture and multi-task
loss design. It does not include the additional techniques used in the
original leaderboard submission — 2018-season pretraining, augmentation,
EMA, test-time augmentation, k-fold cross-validation, and ensembling — which
are out of scope for this project.

## Repository structure

```text
data/train/            # Kaggle CSVs (not committed — see data/README.md)
notebooks/eda.py        # exploratory data analysis
notebooks/01_eda.ipynb
src/features.py          # feature engineering
src/preprocess.py        # builds fixed-size (T, 22, F) tensors per play
src/model.py             # STTransformer: dual temporal/spatial attention
src/losses.py            # main + auxiliary losses
src/train.py             # training loop, validation, checkpointing
src/baseline.py          # stationary / constant-velocity baselines
src/plot_training.py     # training/validation curves
src/plot_trajectory.py   # per-player trajectory visualization
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the Kaggle `input_2023_wXX.csv` / `output_2023_wXX.csv` files under
`data/train/` (see `data/README.md`).

## Usage

```bash
# 1. Exploratory data analysis
python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda

# 2. Build a processed training set
python src/preprocess.py --data-dir data/train --weeks 1 --window 20 \
    --max-plays 1000 --output data/processed/week1_1000.pkl

# 3. Baselines (stationary / constant-velocity)
python src/baseline.py --data data/processed/week1_1000.pkl

# 4. Train
python src/train.py --data data/processed/week1_1000.pkl --epochs 10 \
    --batch-size 8 --hidden 192 --layers 2 --output checkpoints/model.pt

# 5. Training curves
python src/plot_training.py --history outputs/training_history.csv \
    --output outputs/training_curves.png

# 6. Visualize a prediction
python src/plot_trajectory.py --data data/processed/week1_1000.pkl \
    --checkpoint checkpoints/model.pt --index 0 --player-slot 0
```

Scale to more weeks/plays by passing additional `--weeks` and a larger
`--max-plays` to `src/preprocess.py`.
