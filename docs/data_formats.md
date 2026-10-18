# Data formats and evaluation

All paths are resolved from the working directory. Run the provided commands
from the repository root. A YAML contains a nonempty `task_dataset` mapping;
each entry names a task and supplies `root_path`, `data_path`, `seq_len`, and
the appropriate dimensions. Unsupported formats and missing files raise errors.

## Forecasting and imputation: CSV

The CSV has a header and finite numeric observations. `features: M` selects
the configured `columns` list or all columns except `date_column` (default
`date`). `features: S` selects only `target` (default `OT`). Target-only `MS`
forecasting is not implemented. `enc_in` must match the selected channel count.

`split_borders: [train_end, val_end]` gives absolute row offsets, excluding the
header. It takes priority over `split_ratio: [0.7, 0.1, 0.2]`. Rows are split
chronologically; no automatic dataset-specific borders are inferred from a
filename. Forecasting validation/test inputs may include preceding history,
but their prediction targets remain within the respective split. Imputation
windows stay entirely inside their split.

With `scale: true`, means and standard deviations are fit only on the training
split and reused for validation/test. Constant channels use a scale of one.
MSE/MAE are reported in that standardized space. Set `scale: false` when inputs
have already been normalized using an explicitly documented training-only rule.

Imputation inputs must contain complete reference values: the loop creates
missing entries with `--mask_rate`, replaces those inputs with zero, and computes
loss and metrics only on the masked targets. This benchmark-style evaluation
does not infer ground truth for naturally missing CSV cells. Training masks
vary across batches; evaluation uses a seeded mask stream.

The evaluation loop also fixes its frequency-sampling stream separately for
validation and test inside `fork_rng`. On return, the preceding training RNG
state is restored. Keep evaluation batch order and configuration unchanged
when checking repeatability.

## Classification: NPZ

Prepare these six arrays without Python object/pickle arrays:

| Key | Shape and contents |
| --- | --- |
| `x_train`, `x_val`, `x_test` | Finite floating-point `[samples, seq_len, channels]`. |
| `y_train`, `y_val`, `y_test` | Integer `[samples]`, using the same zero-based class mapping in every split. |

All input splits must share the same sequence length and channel count;
`num_class >= 2` bounds the class labels. `scale: true` fits each channel's
normalization over the training sample and time dimensions. Evaluation reports
accuracy as a percentage over the complete test split.

Raw UEA/UCR `.ts` files and variable-length records require external conversion.
Preserve the source train/test protocol, form validation data from training
data, and record padding, resampling, normalization, and class-mapping choices.
The repository does not silently resize records or create validation labels.

## Anomaly detection: NPZ

| Key | Shape and contents |
| --- | --- |
| `train`, `val`, `test` | Finite floating-point `[time, channels]` with a common channel count. |
| `test_labels` | Binary `[test_time]`, where 1 denotes an anomalous point. |
| `train_labels`, `val_labels` | Optional binary point labels; not used to choose the threshold. |

The loader uses training-only normalization and windows of `seq_len` rows.
`window_stride` must not exceed `seq_len`; the final window covers the series
tail. Reconstruction error is averaged over channels, then over overlapping
windows at each time point. The threshold is the
`1 - anomaly_ratio / 100` quantile of validation scores. Test labels are used
only for scoring. Precision, recall, and F1 are pointwise percentages, with no
point adjustment. This explicit protocol may differ from another benchmark's
published threshold or event-adjustment convention.

Raw MSL/SMAP/SMD/SWaT directories are not automatically converted. Preserve
series boundaries and align test labels when preparing an NPZ. Avoid joining
unrelated machines or sessions into a single continuous series merely to fit
the file format.

## Sampling and multiple tasks

`--subsample_pct` is a fraction in `(0, 1]`. It selects training windows or
classification records without replacement using the run seed; validation and
test sets are unchanged. With subsampling enabled, normalization is fitted on
the selected training records or the union of time points covered by selected
training windows. Forecast windows include their training prediction targets
in this fitting set. The same seeded selection supplies normalization statistics
for validation and test. Because windows overlap, a 20% window subset need not
cover only 20% of distinct time points; record the sampling unit when comparing
few-shot protocols.

The multi-task loop visits task loaders round-robin until every loader is
exhausted. Longer loaders therefore contribute more optimization batches. There
are no task-specific loss weights in this entry point. Validation checkpoint
selection averages the per-task validation losses with equal task weights;
loss scales can differ across task types. Configure a suitable task set and
evaluate this choice for the intended experiment.

The `dataset` identity determines prompt sharing: several tasks referring to
the same dataset reuse that dataset's prompt parameters. Changing a task's
display name alone does not create a new prompt if `dataset` is unchanged.
