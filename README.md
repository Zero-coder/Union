# Union

Union is a unified time-series modeling framework for forecasting, imputation, anomaly detection, and classification under heterogeneous temporal dynamics.

This repository is a clean implementation scaffold for the paper:

> Union: Unified Time-Series Modeling for Diverse Tasks under Heterogeneous Temporal Dynamics

The engineering layout is inspired by [mims-harvard/UniTS](https://github.com/mims-harvard/UniTS), but the code here is organized around the Union paper design: unified tokenization, frequency-aware interaction, TCI-MoE sequence operators, and lightweight task heads.

## Highlights

- One shared backbone for generative and predictive time-series tasks.
- Unified token representation with series, prompt, generation, and class tokens.
- Frequency-aware channel and temporal interaction blocks.
- TCI-MoE experts for length-adaptive temporal transformations.
- Minimal runnable training entry point with a synthetic smoke-test dataset.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/smoke_forecast.yaml
```

The default config uses synthetic data so the pipeline can be checked before real datasets are wired in.

## Tasks

- `forecast`: predict future time steps from historical observations.
- `impute`: reconstruct missing values.
- `anomaly`: reconstruct the input sequence and score reconstruction error.
- `classify`: predict a class label from the shared CLS token.

## Status

This is the initial research-code scaffold. The next step is to connect paper benchmark datasets and reproduce the reported tables.
