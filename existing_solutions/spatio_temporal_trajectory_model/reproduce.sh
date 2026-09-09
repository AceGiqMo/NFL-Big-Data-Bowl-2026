#!/usr/bin/env bash
# reproduce.sh — one command per reported number.
# Run from anywhere: the script cd's into its own directory
# (source/existing_solutions/spatio_temporal_trajectory_model/).
set -euo pipefail
cd "$(dirname "$0")"

#python3.14 -m pip install --upgrade pip --user --break-system-packages
#python3.14 -m pip install -r requirements.txt --break-system-packages

# 0) dataset statistics (figures/eda.png)
python3.14 train.py eda

# 1) baseline on the test split (same harness as the model -> fair comparison)
python3.14 train.py baselines

# 2) multi-seed training + evaluation (fixed seeds => reproducible numbers)
for s in 42 43 44; do
  python3.14 train.py train    --seed "$s" --exp-name "seed_$s"
  python3.14 train.py evaluate --exp-name "seed_$s"
done

# 3) aggregate seeds: mean +/- std (figures/seed_summary.png, outputs/seed_summary.json)
python3.14 train.py summarize --exp-names seed_42,seed_43,seed_44

echo "Done. Figures: figures/<exp>/*.png | Metrics: outputs/<exp>/test_results.json"