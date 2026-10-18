"""Single-process supervised training and complete split evaluation."""
import importlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
import yaml

from data_provider.data_factory import data_provider
from union.data import canonical_task
from utils.losses import masked_mse


def read_task_data_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    tasks = config.get("task_dataset") if isinstance(config, dict) else None
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("YAML must contain a nonempty task_dataset mapping.")
    return tasks


def get_task_data_config_list(task_data_config, default_batch_size=None):
    result = []
    for name, original in task_data_config.items():
        config = dict(original)
        if default_batch_size is not None:
            config.setdefault("max_batch", default_batch_size)
        result.append([name, config])
    return result


def iter_task_batches(loaders):
    """Round-robin all batches, including the longer task loaders."""
    active = [(i, iter(loader)) for i, loader in enumerate(loaders)]
    while active:
        next_active = []
        for task_id, iterator in active:
            try:
                batch = next(iterator)
            except StopIteration:
                continue
            yield batch, task_id
            next_active.append((task_id, iterator))
        active = next_active


def make_missing_mask(x, rate, generator=None):
    if not 0 < rate < 1:
        raise ValueError("Mask rate must be strictly between zero and one.")
    mask = torch.rand(x.shape, generator=generator, device="cpu") < rate
    flat = mask.reshape(mask.shape[0], -1)
    # Guarantee that every sequence contributes a hidden-value target.
    flat[~flat.any(dim=1), 0] = True
    return mask.to(x.device)


class Exp_All_Task:
    pretrain = False

    def __init__(self, args):
        self.args = args
        requested = getattr(args, "device", "auto")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable.")
        self.task_data_config = read_task_data_config(args.task_data_config_path)
        self.task_data_config_list = get_task_data_config_list(self.task_data_config, args.batch_size)
        self._data_cache = {}
        self.model = self._build_model()
        if getattr(args, "pretrained_weight", None) and getattr(args, "is_training", 1):
            self._load_checkpoint(args.pretrained_weight, transfer=True)

    def _build_model(self):
        module = importlib.import_module("models." + self.args.model)
        return module.Model(self.args, self.task_data_config_list, pretrain=self.pretrain).to(self.device)

    def _get_data(self, flag):
        if flag not in self._data_cache:
            self._data_cache[flag] = [data_provider(self.args, config, flag)
                                      for _, config in self.task_data_config_list]
        return self._data_cache[flag]

    def _mask_rate(self, training):
        if not self.pretrain:
            return self.args.mask_rate
        low, high = self.args.min_mask_ratio, self.args.max_mask_ratio
        return low + torch.rand(()).item() * (high - low) if training else (low + high) / 2

    def _batch(self, batch, task_id, training=False, generator=None):
        x = batch["x"].to(self.device)
        y = batch["y"].to(self.device)
        task_name = self.task_data_config_list[task_id][1]["task_name"]
        task = "impute" if self.pretrain else canonical_task(task_name)
        missing = None
        model_input = x
        if task == "impute":
            y = x
            missing = make_missing_mask(x, self._mask_rate(training), generator)
            model_input = x.masked_fill(missing, 0.0)
        outputs = self.model(x_enc=model_input, task_id=task_id, task_name=task_name)
        if task == "classify":
            pred = outputs["logits"]
            if pred.ndim != 2 or len(pred) != len(y):
                raise ValueError("Classification output must be [batch, classes].")
            loss = nn.functional.cross_entropy(pred, y)
        else:
            pred = outputs["forecast" if task == "forecast" else "reconstruction"]
            if pred.shape != y.shape:
                raise ValueError("Output %s does not match target %s; check task-specific model configuration."
                                 % (tuple(pred.shape), tuple(y.shape)))
            loss = masked_mse(pred, y, missing) if missing is not None else nn.functional.mse_loss(pred, y)
        return loss, pred, y, missing

    def _checkpoint_path(self, setting, name="checkpoint.pth"):
        return Path(self.args.checkpoints) / setting / name

    def _load_checkpoint(self, path, transfer=False):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError("Checkpoint does not exist: %s" % path)
        payload = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(payload, dict) or "model" not in payload or "model_config" not in payload:
            raise ValueError("Checkpoint needs model and model_config metadata from this implementation; older unversioned weights are not supported.")
        state = payload["model"]
        current_config = asdict(self.model.model.config)
        architecture_keys = ["seq_len", "patch_len", "stride", "d_model", "depth", "n_heads",
                             "num_experts", "top_k", "prompt_num", "kernel_tau", "max_freq_modes"]
        for key in architecture_keys:
            if payload["model_config"].get(key) != current_config[key]:
                raise ValueError("Checkpoint architecture differs at %s. Match the source architecture; use reference_seq_len for its base length." % key)
        current = self.model.state_dict()
        for name, value in state.items():
            if name.endswith("kernel_tau") and name in current and not torch.equal(value, current[name]):
                raise ValueError("Checkpoint kernel_tau buffer does not match the configured fixed kernel.")
        if not transfer:
            self.model.load_state_dict(state, strict=True)
            return
        compatible, skipped = {}, []
        for name, value in state.items():
            adaptable = "prompt" in name or "classification_head" in name
            if name not in current or current[name].shape != value.shape:
                if not adaptable:
                    raise ValueError("Pretrained backbone differs at %s; use a matching architecture." % name)
                skipped.append(name)
            else:
                compatible[name] = value
        missing = [name for name in current if name not in compatible]
        if any("prompt" not in name and "classification_head" not in name for name in missing):
            raise ValueError("Pretrained checkpoint is missing shared-model parameters.")
        self.model.load_state_dict(compatible, strict=False)
        print("Loaded %d tensors; initialized %d target-specific tensors; skipped %d source tensors."
              % (len(compatible), len(missing), len(skipped)))

    def _learning_rate(self, epoch):
        warmup = self.args.warmup_epochs
        if warmup and epoch < warmup:
            return self.args.learning_rate * (epoch + 1) / warmup
        if getattr(self.args, "lradj", "cosine" if self.pretrain else "supervised") == "constant":
            return self.args.learning_rate
        floor = self.args.min_lr if self.args.min_lr is not None else 0.0
        progress = (epoch - warmup) / max(self.args.train_epochs - warmup - 1, 1)
        return floor + (self.args.learning_rate - floor) * (1 + math.cos(math.pi * progress)) / 2

    def train(self, setting):
        train_data = self._get_data("train")
        self._get_data("val")
        loaders = [loader for _, loader in train_data]
        total_batches = sum(len(loader) for loader in loaders)
        if not total_batches:
            raise ValueError("No training batches.")
        destination = self._checkpoint_path(setting)
        destination.parent.mkdir(parents=True, exist_ok=True)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                weight_decay=self.args.weight_decay,
                                betas=(0.9, getattr(self.args, "beta2", 0.999)),
                                eps=getattr(self.args, "eps", 1e-8))
        best, history = math.inf, []
        for epoch in range(self.args.train_epochs):
            self.model.train()
            for group in optimizer.param_groups:
                group["lr"] = self._learning_rate(epoch)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for step, (batch, task_id) in enumerate(iter_task_batches(loaders)):
                loss, _, _, _ = self._batch(batch, task_id, training=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss.")
                group_start = (step // self.args.acc_it) * self.args.acc_it
                accumulation_size = min(self.args.acc_it, total_batches - group_start)
                (loss / accumulation_size).backward()
                if (step + 1) % self.args.acc_it == 0 or step + 1 == total_batches:
                    if self.args.clip_grad is not None:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(loss.item())
            validation = self.evaluate("val")
            val_loss = float(np.mean([item["loss"] for item in validation.values()]))
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "val_loss": val_loss})
            payload = {"format_version": 2, "model": self.model.state_dict(), "optimizer": optimizer.state_dict(),
                       "epoch": epoch + 1, "seed": self.args.fix_seed, "args": vars(self.args),
                       "task_config": self.task_data_config, "model_config": asdict(self.model.model.config),
                       "validation": validation}
            torch.save(payload, self._checkpoint_path(setting, "last.pth"))
            if val_loss < best:
                best = val_loss
                torch.save(payload, destination)
            print("Epoch %d: train_loss=%.6f val_loss=%.6f" % (epoch + 1, np.mean(losses), val_loss))
        self._load_checkpoint(destination)
        self._checkpoint_path(setting, "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        return self.model

    def _point_scores(self, task_id, flag):
        dataset, loader = self._get_data(flag)[task_id]
        if not hasattr(dataset, "values"):
            raise ValueError("Pointwise anomaly evaluation requires an un-subsampled WindowDataset.")
        sums = np.zeros(len(dataset.values), dtype=np.float64)
        counts = np.zeros(len(dataset.values), dtype=np.int64)
        for batch in loader:
            _, pred, target, _ = self._batch(batch, task_id)
            errors = (pred - target).square().mean(dim=-1).cpu().numpy()
            for start, row in zip(batch["start"].tolist(), errors):
                sums[start:start + len(row)] += row
                counts[start:start + len(row)] += 1
        if not counts.all():
            raise ValueError("Anomaly windows leave unevaluated time points.")
        return sums / counts

    @torch.no_grad()
    def evaluate(self, flag):
        # Replay stochastic frequency selection without consuming training RNG state.
        with torch.random.fork_rng():
            torch.manual_seed(int(self.args.fix_seed or 0) + {"val": 100000, "test": 200000}[flag])
            return self._evaluate(flag)

    def _evaluate(self, flag):
        self.model.eval()
        results = {}
        for task_id, ((name, config), (dataset, loader)) in enumerate(zip(self.task_data_config_list, self._get_data(flag))):
            task = "impute" if self.pretrain else canonical_task(config["task_name"])
            generator = torch.Generator().manual_seed(int(self.args.fix_seed or 0) + 1000 + task_id)
            loss_sum = count = correct = squared = absolute = elements = 0
            for batch in loader:
                loss, pred, target, mask = self._batch(batch, task_id, generator=generator)
                size = len(target)
                loss_sum += loss.item() * size
                count += size
                if task == "classify":
                    correct += (pred.argmax(dim=-1) == target).sum().item()
                else:
                    difference = pred - target
                    if mask is not None:
                        difference = difference[mask]
                    squared += difference.square().sum().item()
                    absolute += difference.abs().sum().item()
                    elements += difference.numel()
            item = {"loss": loss_sum / count, "samples": count}
            if task == "classify":
                item["accuracy_percent"] = 100.0 * correct / count
            else:
                item.update(loss=squared / elements, mse=squared / elements,
                            mae=absolute / elements, evaluated_elements=elements)
            if task == "anomaly" and flag == "test":
                # Select the threshold from unlabeled validation scores, never test labels.
                calibration = self._point_scores(task_id, "val")
                threshold = float(np.quantile(calibration, 1 - self.args.anomaly_ratio / 100.0))
                scores = self._point_scores(task_id, "test")
                labels = dataset.point_labels.astype(bool)
                predictions = scores > threshold
                tp = int(np.sum(predictions & labels))
                fp = int(np.sum(predictions & ~labels))
                fn = int(np.sum(~predictions & labels))
                precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                item.update(precision_percent=100 * precision, recall_percent=100 * recall,
                            f1_percent=100 * 2 * tp / max(2 * tp + fp + fn, 1),
                            threshold=threshold, test_points=len(scores), point_adjustment=False)
            results[name] = item
        return results

    def test(self, setting, load_pretrain=False):
        if load_pretrain:
            self._load_checkpoint(self.args.pretrained_weight or self._checkpoint_path(setting))
        results = self.evaluate("test")
        destination = self._checkpoint_path(setting, "test_metrics.json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"seed": self.args.fix_seed, "tasks": results}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2))
        return results
