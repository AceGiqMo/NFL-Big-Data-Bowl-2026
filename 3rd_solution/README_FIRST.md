# Start here

This folder is the university-course reproduction of the core ideas from the NFL Big Data Bowl 2026 3rd-place solution.

## 1. Put Kaggle data locally

Place `input_2023_w01.csv` and `output_2023_w01.csv` in `data/train/`.

Do not commit these CSVs to GitHub.

## 2. Create the environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run EDA

```powershell
python notebooks/eda.py --data-dir data/train --week 1 --output-dir outputs/eda
```

## 4. Prepare a small real dataset

```powershell
python src/preprocess.py --data-dir data/train --weeks 1 --window 20 --max-plays 5000 --output data/processed/week1_5000.pkl
```

## 5. Evaluate simple baselines

```powershell
python src/baseline.py --data data/processed/week1_5000.pkl
```

## 6. Train the reproduced model

Start with a smoke test:

```powershell
python src/train.py --data data/processed/week1_5000.pkl --epochs 3 --batch-size 8 --hidden 128 --layers 1 --output checkpoints/smoke.pt
```

Then use the course experiment:

```powershell
python src/train.py --data data/processed/week1_5000.pkl --epochs 10 --batch-size 8 --hidden 192 --layers 2 --output checkpoints/third_place_minimal.pt
```

Training also writes `outputs/training_history.csv`.

Plot the curves with:

```powershell
python src/plot_training.py --history outputs/training_history.csv --output outputs/training_curves.png
```

## 7. Plot one predicted trajectory

```powershell
python src/plot_trajectory.py --data data/processed/week1_5000.pkl --checkpoint checkpoints/third_place_minimal.pt --index 0 --player-slot 0 --output outputs/prediction_trajectory.png
```

## 8. What to show in the report

- EDA: dataset structure, player/target counts, horizon distribution and one play visualization.
- Baselines: stationary and constant-velocity RMSE.
- Model: training loss and validation RMSE by epoch.
- Prediction: ground-truth vs predicted trajectory for a target player.
- Interpretation: lower RMSE means the predicted field position is closer to the true position; the unit is yards.

## What this is and is not

This is a transparent educational reproduction of the documented core architecture. It is not a byte-for-byte or leaderboard-identical reproduction of the original 3rd-place submission, which also used additional pretraining, augmentation, EMA, TTA, cross-validation and ensembling.
