# Reproduction Gap Against fecam

This checklist tracks what Union still needs before it matches the engineering coverage of `mims-harvard/fecam`.

## Added In This Scaffold

- fecam-style supervised entry point: `run.py`.
- fecam-style masked reconstruction pretraining entry point: `run_pretrain.py`.
- Experiment modules under `exp/`.
- Multi-task config files under `data_provider/`.
- Balanced task iterator and DDP utility stubs under `utils/`.
- Script families for supervised, pretraining + prompt tuning, few-shot, and zero-shot runs.
- `models/Union.py` adapter so experiment code can import `--model Union`.

## Still Missing For Full Paper Reproduction

- Real CSV loaders for ETT, ECL, Exchange, Weather, ILI, and other forecasting datasets.
- UEA/UCR `.ts` classification loader and collate function.
- Segment loaders for MSL, PSM, SMAP, SMD, and SWAT anomaly detection.
- Imputation masking policy parity with the benchmark scripts.
- Checkpoint loading modes for supervised finetuning, prompt tuning, and zero-shot evaluation.
- Distributed training parity with fecam DDP/AMP behavior.
- Metric tables and result aggregation scripts for every paper table.
