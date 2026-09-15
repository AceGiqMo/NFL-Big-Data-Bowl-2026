# NFL Trajectory Prediction — Reproduction Study

This repository contains a notebook-based implementation for the **NFL Big Data Bowl 2026 Prediction** trajectory forecasting task.

The project is intentionally kept as a single notebook rather than split into Python modules. The notebook contains the complete preprocessing, feature engineering, model definition, training procedure, validation, artifact creation, and evaluation workflow.

## Original sources

- [First-place solution write-up](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/writeups/public-3rd-solution)
- [Original training notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-train)
- [Original inference notebook](https://www.kaggle.com/code/chack3/nfl2026-1st-place-inference)

## Repository structure

```text
nfl_trajectory_model/
├── checkpoints/
│   ├── nfl_1st_place_best.pt
│   └── nfl_1st_place_model.pkl
├── figures/
│   ├── rmse_by_frame.png
│   ├── trajectories.png
│   ├── training_curves.png
│   └── error_hist.png
├── notebooks/
│   └── solution.ipynb
└── README.md
```

## Task

The goal is to predict the future trajectories of NFL players from the observed part of a play.

The model receives a fixed-size representation of the observed play and predicts future x/y displacements for the players that require predictions. The implementation uses both dynamic temporal features and static player/play features.

The prediction horizon is padded to a maximum of **48 frames**, while the observed input is represented using the most recent **20 frames**.

## Approach

The notebook implements the following pipeline:

1. Load weekly NFL input/output data.
2. Remove known problematic plays and plays without a passer.
3. Normalize play direction to a common coordinate system.
4. Represent each play as structured player-by-time arrays.
5. Apply training-time spatial and temporal augmentations.
6. Construct dynamic and static features.
7. Normalize features and targets using precomputed training statistics.
8. Pad players and sequences to fixed dimensions.
9. Create five deterministic game-level folds with `GroupKFold` and train one explicitly selected holdout fold (`VALID_FOLD`).
10. Train a neural trajectory model.
11. Use Gaussian NLL for probabilistic coordinate prediction.
12. Add an auxiliary velocity/acceleration prediction objective.
13. Maintain an exponential moving average (EMA) of model weights.
14. Select and preserve the best EMA checkpoint according to holdout RMSE.
15. Evaluate the model in absolute field coordinates.
16. Produce diagnostic plots.

## Model architecture

The model combines several components:

- depthwise temporal convolution in the sequence stem;
- residual convolutional blocks;
- a static-feature stem;
- feature fusion;
- a multi-layer Transformer encoder;
- separate prediction heads for the mean and variance of the main coordinate prediction;
- auxiliary heads for velocity and acceleration.

The main prediction is probabilistic: the network predicts both a mean and a variance, which are used by the Gaussian negative log-likelihood loss.

The auxiliary task predicts four quantities per future frame:

- x velocity;
- y velocity;
- x acceleration;
- y acceleration.

## Difference from the original first-place solution

This repository should be treated as a **reproduction attempt of the first-place approach, not an exact reproduction of the original submission**.

The implementation follows the same general modeling direction documented in the notebook, including:

| Component | This implementation | First-place approach referenced in the notebook |
|---|---|---|
| Temporal representation | Recent 20 observed frames | Similar sequence-based representation |
| Feature types | Position, orientation, speed-related directional features, ball landing and player-relative features | Same general feature-engineering strategy |
| Spatial augmentation | Rotation, vertical flip, x/y shifts | Used as part of the referenced approach |
| Temporal augmentation | Frame shift | Used as part of the referenced approach |
| Core architecture | Convolutional stem + residual blocks + Transformer + prediction heads | Transformer/convolutional trajectory architecture |
| Main objective | Gaussian NLL | Probabilistic trajectory prediction |
| Auxiliary objective | Velocity + acceleration | Auxiliary motion prediction |
| EMA | Custom `ModelEMA` implementation | Referenced implementation uses an EMA utility equivalent to `timm`'s model EMA |
| Data split | One explicitly selected holdout fold from a 5-part `GroupKFold` | Original used repeated 5-fold CV; this notebook intentionally trains one model |
| Training | 35 epochs with training-set repetition | Training schedule follows the settings available in the notebook, but is not a byte-for-byte recreation of the original pipeline |
| Artifacts | `.pt` checkpoint and `.pkl` containing weights/configuration/statistics | Original submission artifacts are not identical to these files |

### Why the first-place solution was not reproduced exactly

An exact reproduction cannot be claimed because the notebook does not contain the complete original first-place training code and execution environment.

Instead, the implementation reconstructs the main ideas and settings that were available from the reference solution. Several details are therefore implemented independently:

- the EMA mechanism is implemented locally rather than relying on the original external EMA implementation;
- preprocessing and play filtering are explicitly reconstructed in the notebook;
- normalization statistics are stored directly in the notebook;
- the validation setup uses a local `GroupKFold` split;
- model artifacts are packaged in a custom `.pkl` structure;
- the original competition environment, exact training pipeline, and all original implementation details are not available as part of this repository.

Consequently, the resulting model should be interpreted as a **faithful reproduction attempt at the method level**, rather than a reproduction of the exact first-place submission.

## Limitations

### 1. Not an exact reproduction

The most important limitation is that the implementation cannot guarantee identical behavior to the original first-place solution. Matching the architecture at a high level does not guarantee identical optimization dynamics or predictions.

### 2. Hard-coded normalization statistics

Feature and target normalization statistics are stored as precomputed constants. Recomputing them on a different dataset split may produce different results.

### 3. Computational constraints

The original competition solution may have relied on a different computational budget and execution environment. Training time, batch processing, hardware availability, and numerical behavior can affect the final model.

### 4. Reconstructed artifacts

The `.pt` and `.pkl` files included in the repository are artifacts generated by this implementation. They are not claimed to be the original first-place submission files.

### 5. Competition-specific preprocessing

Several known play exclusions and coordinate transformations are encoded directly in the notebook. These decisions are specific to the reproduction and should be reviewed if the code is transferred to another dataset or competition version.

## Output artifacts

### `nfl_1st_place_best.pt`

PyTorch state dictionary containing the best EMA weights selected by holdout RMSE. The final non-EMA weights do not overwrite this checkpoint.

### `nfl_1st_place_model.pkl`

A serialized artifact containing:

- model state dictionary;
- model configuration;
- dynamic feature definitions;
- feature normalization statistics;
- target normalization statistics;
- validation setup and the best holdout RMSE.

The `.pkl` packages exactly the same best EMA state dictionary as the `.pt` file, making it possible to reconstruct the selected model without rerunning the training stage. The equality of all tensors in the two checked-in artifacts was verified after the latest run.

## Latest run summary

The current artifacts were trained with `VALID_FOLD = 0`. The best EMA checkpoint reached a holdout RMSE of **0.4538 yards**. This is a score for one game-level holdout fold, not the mean of a complete five-fold cross-validation run.

## Figures

### `training_curves.png`

Shows the evolution of:

- training Gaussian NLL;
- validation Gaussian NLL;
- holdout RMSE for the explicitly selected fold.

Training Gaussian NLL falls from approximately **-0.48** to **-1.15**, while validation NLL improves from approximately **-1.16** at the first evaluation to **-1.32** near the end of training. Holdout RMSE decreases from **0.5200** to its minimum of **0.4538** around epoch 27, then fluctuates within a narrow range through epoch 35. This indicates stable convergence with only a small late-stage plateau rather than pronounced overfitting. The selected EMA checkpoint preserves the minimum instead of using the final epoch automatically.

Since the notebook trains only `VALID_FOLD`, the red curve must not be interpreted as a mean across all five folds.

### `rmse_by_frame.png`

Shows holdout RMSE separately for every future frame that actually has target samples. Frames with zero observations are omitted instead of being displayed as zero-error predictions.

The error is very small over the first few frames, reaches roughly **0.3 yards** by frame 10, exceeds **1 yard** around frames 18–19, and rises to approximately **2.7–3.0 yards** over the longest observed horizons. The overall trend confirms that uncertainty accumulates with prediction distance. Small local decreases around frames 29–32 should not be interpreted as improved long-range forecasting because the number and composition of available samples change at each horizon. Fold 0 contains targets through frame 40; later frames are therefore absent from the chart.

### `error_hist.png`

Shows the distribution of Euclidean prediction errors across valid player-frame pairs for the reloaded best EMA checkpoint. The distribution is strongly right-skewed: the median error is **0.13 yards**, most samples are concentrated close to zero, and a small number of difficult player-frame cases form a long tail extending beyond 10 yards. Consequently, aggregate RMSE is influenced substantially more by rare large misses than the median error suggests.

### `trajectories.png`

Contains six deterministic holdout examples sampled as individual `(play, player)` pairs. Every subplot shows exactly one player's ground-truth and predicted path in absolute field coordinates and marks the last observed position with a star. Trajectories belonging to different players are never flattened or connected.

The examples show that the model usually captures the initial direction and curvature accurately. Shorter trajectories such as samples `#1835`, `#1234`, and `#256` remain close to the ground truth. Errors accumulate on longer trajectories: sample `#2151` overshoots the player's final position, sample `#2395` finishes too far to the right and too high, and sample `#1249` develops an unrealistic late reversal. These cases are consistent with the horizon-dependent RMSE and identify long-range motion as the main remaining weakness.


## Reproducibility

The notebook fixes the main random seeds and stores the preprocessing statistics and model configuration required by the training/evaluation pipeline.

To reproduce the experiment, attach the NFL Big Data Bowl 2026 Prediction competition data, enable a Kaggle T4 GPU, and execute the cells sequentially. The current implementation uses one GPU.
