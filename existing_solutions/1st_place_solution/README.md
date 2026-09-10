# NFL Trajectory Prediction — Reproduction Study

This repository contains a notebook-based implementation for the **NFL Big Data Bowl 2026 Prediction** trajectory forecasting task.

The project is intentionally kept as a single notebook rather than split into Python modules. The notebook contains the complete preprocessing, feature engineering, model definition, training procedure, validation, artifact creation, and evaluation workflow.

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
9. Split plays with `GroupKFold` using `game_id` as the grouping variable.
10. Train a neural trajectory model.
11. Use Gaussian NLL for probabilistic coordinate prediction.
12. Add an auxiliary velocity/acceleration prediction objective.
13. Maintain an exponential moving average (EMA) of model weights.
14. Select the best checkpoint according to validation CV-RMSE.
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
| Data split | One 5-fold `GroupKFold` split, with the first generated split used | Original training/validation procedure is not fully reproduced here |
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

PyTorch state dictionary containing the selected model weights.

### `nfl_1st_place_model.pkl`

A serialized artifact containing:

- model state dictionary;
- model configuration;
- dynamic feature definitions;
- feature normalization statistics;
- target normalization statistics.

This makes it possible to reconstruct the model without rerunning the training stage.

## Figures

### `training_curves.png`

Shows the evolution of:

- training Gaussian NLL;
- validation Gaussian NLL;
- validation CV-RMSE.

In this run training NLL decreased monotonically from **-0.4801** at epoch 1 to **-1.1460** at epoch 35, and validation NLL kept improving throughout the schedule, reaching **-1.3241** at the last epoch with no sign of overfitting. Validation was enabled at epoch 6 (`START_VALID_EPOCH = 6`); the EMA-weighted model reached its best CV-RMSE of **0.4538 yards at epoch 27**, after which the metric entered a narrow plateau in the 0.453–0.456 range for the remaining 8 epochs. The full-epoch validation CV-RMSE declined from **0.5200** yards at epoch 6 to **0.4561** yards at epoch 35 — a relative improvement of about 12.3% over the first validation epoch. Each epoch took ~122 s on a single GPU.

The plot is useful for checking optimization progress, detecting divergence, and observing whether validation performance improves as training proceeds.

### `rmse_by_frame.png`

Shows RMSE separately for each future prediction frame (1 to 48).

For a trajectory-forecasting head the per-frame RMSE grows roughly monotonically with the prediction horizon: short-horizon frames sit well below the aggregate CV-RMSE (the first frame is essentially a one-step extrapolation and is on the order of 10⁻² yards), while the latest predicted frames reach the upper end of the curve at the multi-yard level. The growth reflects the compounding uncertainty of player motion — small angular or velocity errors at the input frame accumulate into larger positional deviations the further into the future the model projects. The aggregate validation CV-RMSE (joint over x and y) is **0.4844 yards**, with per-axis values of **0.4975 yards** along x and **0.4710 yards** along y, computed over **110,958** valid (player, future-frame) pairs.

This is particularly important for trajectory forecasting because the difficulty generally depends on the prediction horizon. The plot makes it possible to inspect how prediction error changes as the forecast moves further into the future.

### `error_hist.png`

Shows the distribution of Euclidean prediction errors across valid player-frame pairs.

The distribution is computed over the same **110,958** valid (player, future-frame) pairs from the validation fold. It is strongly right-skewed: the bulk of the mass is concentrated near zero, but a long thin tail extends into the multi-yard range. Concretely:

| Statistic | Error (yards) |
|---|---|
| Mean | 0.354 |
| Median (50%) | 0.146 |
| 75th percentile | 0.391 |
| 90th percentile | 0.876 |
| 95th percentile | 1.346 |
| 99th percentile | 2.843 |
| Max | 14.561 |

Half of all predicted positions land within ~0.15 yards (~13.7 cm) of the ground truth, 75% within ~0.39 yards, and 95% within 1.35 yards. The 99th-percentile value of 2.84 yards and the maximum of 14.56 yards indicate that the remaining large-error cases are dominated by a small number of plays with unusual motion patterns (e.g., extended-scramble / broken-play situations where the targeted receiver deviates sharply from the route implied by the input frames).

The figure complements the aggregate RMSE by showing the typical error range and the presence of large-error cases. The median and percentile statistics are also calculated in the notebook.

### `trajectories.png`

Contains six validation examples comparing ground-truth trajectories with model predictions.

The six examples are sampled with a fixed RNG seed (`np.random.default_rng(42)`) from the validation fold; only plays with at least one valid future frame are eligible. Each subplot overlays the predicted path (× markers, thin line) on the ground-truth path (● markers) in absolute field coordinates, with x on the horizontal axis and y on the vertical axis. Across the six samples the predicted trajectories follow the overall direction and curvature of the ground truth, with deviations starting to accumulate near the tail of the horizon — which is consistent with the per-frame RMSE growth visible in `rmse_by_frame.png`. None of the six sampled plays exhibit catastrophic failure (no path that diverges to a clearly wrong region of the field); the larger end-of-horizon deviations are the rule rather than the exception, and reflect the inherent difficulty of forecasting later frames rather than a systematic bias of the model.

This is a qualitative diagnostic: it shows whether the predicted paths follow the overall movement pattern of the actual trajectories and helps reveal cases where the model deviates substantially from the target path.


## Reproducibility

The notebook fixes the main random seeds and stores the preprocessing statistics and model configuration required by the training/evaluation pipeline.

To reproduce the experiment, place the notebook in the Kaggle environment with the required NFL Big Data Bowl 2026 Prediction dataset and execute the cells sequentially, using GPU T4 x2 accelerator.