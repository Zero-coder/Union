# Union

Union is a shared time-series framework for forecasting, imputation,
classification, and reconstruction-based anomaly detection. Dataset-specific
prompt tokens condition a backbone with frequency-aware interaction and
temporal-consistency mixture-of-experts blocks.

## Model components

- Variable-preserving patch tokens with learned dataset prompts and CLS/GEN tokens.
- Sequential channel and temporal frequency interaction, with shared processing
  of the real and imaginary components within each stage.
- Learnable expert weights resized to the current token count and modulated by a fixed
  exponential temporal kernel; the default decay scale is `tau=0.2`.
- Top-k routing that executes only the selected sample/expert pairs.
- Shared generation and classification modules with task-specific output shapes.

Tensor layouts, frequency selection, routing, and decoding are described in
[`docs/architecture.md`](docs/architecture.md).

## Installation

Use a Python environment with PyTorch configured for your CPU or CUDA device,
then install the repository dependencies:

```bash
pip install -r requirements.txt
```

Run the following commands from the repository root.

## Prepare the data

| Task | Input format | Example configuration |
| --- | --- | --- |
| Forecasting | Chronologically ordered CSV | [`configs/forecast_csv.yaml`](configs/forecast_csv.yaml) |
| Imputation | Chronologically ordered CSV | [`data_provider/imputation.yaml`](data_provider/imputation.yaml) |
| Classification | NPZ containing train/validation/test samples and labels | [`configs/classification_npz.yaml`](configs/classification_npz.yaml) |
| Anomaly detection | NPZ containing train/validation/test series and test labels | [`data_provider/anomaly_detection.yaml`](data_provider/anomaly_detection.yaml) |
| Joint training | Multiple task entries with their own input files | [`data_provider/multi_task.yaml`](data_provider/multi_task.yaml) |

Set each dataset's path, split boundaries, variables, and task parameters in
the YAML. Missing files and invalid shapes raise an error. CSV normalization
uses training observations; few-shot normalization uses the selected training
subset. The exact fields, array shapes, split rules, and evaluation conventions
are documented in [`docs/data_formats.md`](docs/data_formats.md).

The forecasting example expects `dataset/ETT-small/ETTh1.csv`, with a `date`
column and seven numeric observation columns. It uses 96 input steps, a
96-step prediction horizon, and split borders at rows 8640 and 11520.

## Train and evaluate

```bash
python run.py --is_training 1 --model_id Union_ETTh1 --model Union --task_data_config_path configs/forecast_csv.yaml --fix_seed 2021 --train_epochs 10 --batch_size 32 --learning_rate 1e-4 --min_lr 1e-6 --weight_decay 1e-5
```

Each run saves the best validation checkpoint as `checkpoint.pth`, the final
checkpoint as `last.pth`, and its history and complete-test-split metrics as
`training_history.json` and `test_metrics.json` beneath `checkpoints/`.

To evaluate a saved model, replace `RUN_DIRECTORY` with the directory printed
by training and use matching architecture settings and dataset identities:

```bash
python run.py --is_training 0 --model_id Union_eval --model Union --task_data_config_path configs/forecast_csv.yaml --pretrained_weight checkpoints/RUN_DIRECTORY/checkpoint.pth --fix_seed 2021
```

Checkpoints contain model weights and `model_config` metadata. Loading checks
architecture settings as well as tensor shapes. Older unversioned checkpoints
and bare state dictionaries are incompatible with this interface.

## Pretraining and few-shot adaptation

```bash
bash scripts/pretrain_prompt_learning/Union_pretrain_x128.sh
bash scripts/few_shot_newdata/Union_finetune_few_shot_newdata_pct20.sh checkpoints/RUN_DIRECTORY/checkpoint.pth
```

The first command performs masked reconstruction pretraining. The second loads
compatible shared weights, initializes a prompt for the target dataset, and
jointly fine-tunes the model and target prompt on 20% of the target training
windows. Prepare the target ETTh2 CSV specified in
[`data_provider/fewshot_new_task.yaml`](data_provider/fewshot_new_task.yaml).

If the target input length differs, set `--reference_seq_len` to the source
model's reference length. This preserves the saved positional and expert base
parameter shapes while the target YAML sets the actual input length.

## Reproducibility and tests

Use `--fix_seed` to set the random seed; `--itr` runs use successive seeds.
Randomness includes parameter initialization, data ordering, missing-value
masks, and frequency selection. Validation and test evaluation replay separate
seeded streams without changing the training RNG state. Repeating evaluation
requires the same configuration and batch order. CPU and CUDA results need not
be bitwise identical.

```bash
python -m unittest discover -s tests -v
python -m unittest discover -s analysis/prompt_similarity -p "test_*.py" -v
```

The implementation has passed 26 CPU functional tests covering model gradients,
variable input shapes, routing, masking, data splits, training, checkpoint
loading, and evaluation, plus six prompt-analysis tests. Benchmark accuracy and
GPU scalability have not been evaluated for this release; trained benchmark
checkpoints are not included.

## Figure 5 quantitative analysis

[`analysis/prompt_similarity/`](analysis/prompt_similarity/) contains the
similarity matrix, dataset-domain assignments, metric results, and calculation
script. Its README records the input provenance, metric definitions, and
commands for reproducing the numerical analysis.

## Implementation scope

Training supports a single process, masked reconstruction pretraining, and
full fine-tuning. Prompt-only tuning and zero-shot adaptation are not available.
Classification requires prepared fixed-length NPZ arrays; native UEA/UCR `.ts`
loading and raw anomaly-archive conversion are not included. Unsupported
options fail explicitly. See [`docs/reproduction_gap.md`](docs/reproduction_gap.md)
for the complete implementation and evaluation scope.

Some script filenames are retained for compatibility. The `zero_shot` script
performs ordinary checkpoint evaluation, and `download_data_all.sh` displays
data-preparation instructions. Use the configuration table above to select a
task and consult each script before running it.

## Repository layout

| Directory | Contents |
| --- | --- |
| `union/` | Core model and file-backed dataset readers |
| `models/` | Dataset/task adapter for the shared model |
| `exp/` | Training, pretraining, and full-split evaluation |
| `configs/`, `data_provider/` | Dataset schemas, configurations, and data factory |
| `scripts/` | Task-specific command examples |
| `tests/` | Model and training functional tests |
| `analysis/prompt_similarity/` | Figure 5 numerical analysis |
| `docs/` | Architecture, data formats, and implementation scope |

PyTorch, NumPy, PyYAML, and the datasets retain their respective licenses and
usage conditions.
