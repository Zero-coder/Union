# Union

Union is a time-series framework for forecasting, imputation, classification,
and reconstruction-based anomaly detection. It combines dataset-specific prompt
tokens, frequency-aware interaction, and temporal-consistency mixture-of-experts
blocks in a shared backbone.

This release provides model components, file-backed data readers, supervised
training, masked reconstruction pretraining, checkpoint evaluation, and full
fine-tuning. The examples specify the behavior of this implementation. They are
not a recovery of the original experimental checkpoints or a claim that the
paper's benchmark tables have been reproduced by this release.

## Start with a real CSV

Run commands from the repository root. Install PyTorch for the intended CPU or
CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Place a real ETTh1 file at `dataset/ETT-small/ETTh1.csv`. The example expects a
header, a `date` column, and seven numeric observation columns. Missing files
raise an error; there is no generated-data fallback.

```bash
python run.py --is_training 1 --model_id Union_ETTh1 --model Union --task_data_config_path configs/forecast_csv.yaml --fix_seed 2021 --train_epochs 10 --batch_size 32 --learning_rate 1e-4 --min_lr 1e-6 --weight_decay 1e-5
```

The example uses 96 input steps, a 96-step forecast, and explicit chronological
split borders at rows 8640 and 11520. Edit the YAML for another CSV, column set,
split, or horizon. The supported data formats and evaluation rules are in
[`docs/data_formats.md`](docs/data_formats.md).

Equivalent Bash entry:

```bash
bash scripts/supervised_learning/Union_supervised.sh
```

## Training and evaluation

Each run records `checkpoint.pth` (best validation loss), `last.pth`,
`training_history.json`, and `test_metrics.json` beneath its run directory in
`checkpoints/`. Training uses all configured training batches; checkpoint
selection uses validation loss, and final evaluation covers the complete test
split.

Use an explicit checkpoint path and the same architecture and dataset identity
for evaluation:

```bash
python run.py --is_training 0 --model_id Union_eval --model Union --task_data_config_path configs/forecast_csv.yaml --pretrained_weight checkpoints/RUN_DIRECTORY/checkpoint.pth --fix_seed 2021
```

`RUN_DIRECTORY` is the run directory printed by training. A path of `auto` is
not supported. Checkpoint evaluation loads weights strictly. Fine-tuning with
`--is_training 1 --pretrained_weight PATH` loads compatible shared parameters;
new dataset prompts and incompatible classification-head dimensions are
initialized for the target task. All model parameters are trainable during
fine-tuning.

Checkpoints must include this implementation's `model` and `model_config`
metadata. Architecture settings are checked even when tensor shapes happen to
match, including attention heads, routing, frequency selection, and the fixed
kernel. Older unversioned checkpoints and bare state dictionaries are rejected.

If the target input length differs, set `--reference_seq_len` to the source
model's reference length so its learned positional and expert base matrices
retain their saved shapes. The actual target length still comes from its YAML.

```bash
bash scripts/pretrain_prompt_learning/Union_pretrain_x128.sh
bash scripts/few_shot_newdata/Union_finetune_few_shot_newdata_pct20.sh checkpoints/RUN_DIRECTORY/checkpoint.pth
```

The first script performs masked reconstruction pretraining. The second performs
full fine-tuning on 20% of the target training windows and requires a compatible
source checkpoint. Prepare the ETTh2 CSV before using the target example.
Prompt-only tuning and a zero-shot adaptation protocol are not implemented.

## Configurations

| Configuration | Input and task |
| --- | --- |
| `configs/forecast_csv.yaml` | ETTh1 CSV forecasting; default real-data example. |
| `data_provider/imputation.yaml` | CSV imputation, scored only on masked entries. |
| `configs/classification_npz.yaml` | Fixed-length classification arrays in an explicitly prepared NPZ. |
| `data_provider/anomaly_detection.yaml` | Anomaly NPZ with train/validation/test series and test labels. |
| `data_provider/multi_task.yaml` | Four-task CSV/NPZ example; supply every configured input file. |
| `data_provider/multi_task_pretrain.yaml` | Masked reconstruction pretraining on real observations. |
| `data_provider/fewshot_new_task.yaml` | Target-domain ETTh2 forecasting for full fine-tuning. |

The model examples use a hidden width of 128, three blocks, four attention heads,
patch length and stride 16, ten prompt tokens, four experts with top-2 routing,
up to 64 frequencies per interaction stage, and a fixed kernel scale of 0.2.
These are configurable implementation examples, not a statement that every
published experiment used these settings. See `python run.py --help` for the
available options.

## Randomness and tests

Set `--fix_seed` explicitly. `--itr` runs use successive seeds. Randomness
includes initialization, data ordering, missing-value masks, and frequency
selection when an axis has more than the configured number of frequency bins.
Frequency selection also occurs during evaluation; reproducibility requires
the same seed and execution protocol. The experiment loop uses `fork_rng` and
a fixed validation/test seed stream, then restores the training RNG state.
Repeated evaluations replay that stream when their configuration and batch
order are unchanged. CPU and CUDA execution need not be bitwise identical.

Functional tests cover model shapes and gradients, routing, the fixed kernel,
file reading, masking, checkpoint paths, and evaluation calculations. They use
small controlled fixtures and do not establish benchmark accuracy or GPU
scalability:

```bash
python -m unittest discover -s tests -v
```

The current training entry points support one process. Distributed training,
native UEA/UCR `.ts` loading, and raw anomaly-archive conversion are not provided.
Additional implementation boundaries are listed in
[`docs/reproduction_gap.md`](docs/reproduction_gap.md).

## Repository map

- `union/`: core model and file-backed datasets.
- `models/Union.py`: adapter from configured datasets/tasks to the shared model.
- `exp/`: supervised training, pretraining, and complete-split evaluation.
- `configs/`, `data_provider/`: explicit dataset/configuration examples.
- `scripts/`: command-line entry examples and compatibility wrappers.
- `tests/`: functional model, data, and training checks.
- `analysis/prompt_similarity/`: prompt-structure metric calculation.
- [`docs/architecture.md`](docs/architecture.md): tensor layout and model mechanics.

Existing script filenames are retained for compatibility. In particular,
`configs/smoke_forecast.yaml` now reads real CSV data, and the legacy `zero_shot`
script announces ordinary checkpoint evaluation rather than applying an
unimplemented zero-shot protocol. `download_data_all.sh` prints data-preparation
instructions and does not download or generate datasets.

## Dependencies

The implementation uses PyTorch, NumPy, and PyYAML. Their respective licenses
and notices remain applicable; this update does not replace dependency or
dataset licenses.
