#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Retained as a data-preparation entry point; no benchmark archive is downloaded.
cat <<'DATA_GUIDE'
Prepare the licensed source datasets locally before running Union.

Default forecasting example:
  dataset/ETT-small/ETTh1.csv
  configs/forecast_csv.yaml records the numeric columns and split borders.

Few-shot forecasting example:
  dataset/ETT-small/ETTh2.csv

Classification and anomaly detection require converted NPZ files:
  dataset/classification/classification.npz
  dataset/anomaly/anomaly.npz

Read docs/data_formats.md for required keys, shapes, split rules, and scaling.
Adapt each YAML to the actual channel count and sequence length of your data.
UEA/UCR .ts archives and raw anomaly directories are not converted automatically.
No files were downloaded or fabricated by this command.
DATA_GUIDE
