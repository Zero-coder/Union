# Union

Union is a unified time-series modeling framework for forecasting, imputation, anomaly detection, and classification under heterogeneous temporal dynamics.

This repository is a clean implementation scaffold for the paper:

> Union: Unified Time-Series Modeling for Diverse Tasks under Heterogeneous Temporal Dynamics

The engineering layout is inspired by [Fecam](https://github.com/Zero-coder/Fecam), but the code here is organized around the Union paper design: unified tokenization, frequency-aware interaction, TCI-MoE sequence operators, and lightweight task heads.

## Highlights

- One shared backbone for generative and predictive time-series tasks.
- Unified token representation with series, prompt, generation, and class tokens.
- Frequency-aware channel and temporal interaction blocks.
- TCI-MoE experts for length-adaptive temporal transformations.
- UniTS-style experiment stack for supervised, pretraining, few-shot, imputation, anomaly detection, and zero-shot runs.

## Repository Layout

```text
Union/
  configs/              Example smoke-test configs
  data_provider/        UniTS-style task YAMLs and data factory
  docs/                 Paper and architecture notes
  exp/                  Supervised and pretraining experiment loops
  models/               Experiment-compatible Union model adapter
  scripts/              Reproduction scripts by task family
  utils/                Metrics, losses, scheduling, DDP, dataloader helpers
  run.py                Supervised multi-task entry point
  run_pretrain.py       Masked reconstruction pretraining entry point
  train.py              Minimal smoke-test entry point
  union/                Core Union package
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/smoke_forecast.yaml
```

UniTS-style scripts are also available:

```bash
bash scripts/supervised_learning/Union_supervised.sh
bash scripts/pretrain_prompt_learning/Union_pretrain_x128.sh
bash scripts/few_shot_newdata/Union_finetune_few_shot_newdata_pct20.sh
bash scripts/zero_shot/Union_forecast_new_length_unify.sh
```

## Tasks

- `forecast`: predict future time steps from historical observations.
- `impute`: reconstruct missing values.
- `anomaly`: reconstruct the input sequence and score reconstruction error.
- `classify`: predict a class label from the shared CLS token.

## Status

The repository now has a UniTS-like research-code layout. The current data factory uses synthetic data as a smoke-test fallback; the next milestone is wiring the real benchmark loaders listed in `docs/reproduction_gap.md`.
