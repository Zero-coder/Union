# Implementation coverage and reproduction scope

This release supplies executable model and data paths. Functional correctness
and reproducing the paper's benchmark measurements are separate checks.

## Implemented

- Variable-preserving patch tokens, dataset-specific prompts, CLS/GEN tokens,
  sequential channel/temporal frequency interaction, and selected-bin inverse FFT.
- Fixed exponential temporal kernels with learned expert weights, interpolation,
  and sample-specific sparse expert execution.
- Shared patch reconstruction and task-specific output lengths.
- File-backed CSV forecasting/imputation and NPZ classification/anomaly readers,
  with explicit splits, channel checks, and training-only normalization.
- Supervised multi-task training, masked reconstruction pretraining, full
  fine-tuning from explicit compatible checkpoints, and complete-split testing.
- AdamW, gradient accumulation, gradient clipping, warmup/cosine learning-rate
  options, best-validation checkpoints, and recorded seeds/metrics.
- Analytical tests for model/data/training mechanics and the separate prompt
  structure evaluator.

## Functional validation scope

The tests in `tests/` use small controlled tensors and temporary CSV/NPZ files.
They exercise forward/backward paths, sparse routing, kernel behavior, masking,
split boundaries, data-format checks, and training/evaluation plumbing on CPU.
Run them with:

```bash
python -m unittest discover -s tests -v
```

These checks do not measure the paper's benchmark accuracy, GPU speed/memory,
multi-GPU scaling, or cross-seed statistical significance. No trained benchmark
checkpoints or reproduced paper-result tables are bundled with this update.

## Boundaries to account for

- One training process is supported; distributed training and AMP have not been
  implemented in the experiment entry points.
- Classification requires fixed-length NPZ arrays. Native UEA/UCR `.ts` loading,
  variable-length batching, and raw anomaly-archive conversion are not provided.
- Prompt-only tuning and zero-shot new-domain/offset adaptation are not
  implemented. Legacy options are rejected instead of silently ignored.
- Anomaly evaluation uses a validation-score quantile and pointwise F1 without
  point adjustment. Imputation is scored only on artificially hidden reference
  values. Match these choices explicitly when comparing benchmark protocols.
- Few-shot selection is defined over windows or classification records;
  overlapping windows can cover more distinct time points than their sampled
  fraction suggests. Normalization uses only the selected training data. The
  task iterator visits all batches; it does not balance unequal dataset sizes
  or learn task loss weights.
- Evaluation frequency sampling uses a fixed split-specific RNG stream inside
  `fork_rng`, without consuming the training RNG state. Preserve seeds, batch
  order, and the evaluation protocol when comparing runs.
- Checkpoint loading requires this implementation's `model_config` metadata
  and matching architecture semantics. Bare state dictionaries and earlier
  unversioned checkpoints are rejected. Missing model weights cannot be
  reconstructed from published scores.

## Preparing a benchmark reproduction

Keep the exact dataset versions and preprocessing, split borders, masks,
classification label maps, anomaly threshold rules, training configuration,
software/hardware versions, and seed list with each run. Use real per-seed
metrics for uncertainty estimates and check the resulting tables against the
saved run artifacts. Configuration examples and passed functional tests alone
do not establish equivalence to the original experiment.
