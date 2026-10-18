"""File-backed data readers for the four supported time-series tasks."""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def canonical_task(name):
    aliases = {"forecast": "forecast", "long_term_forecast": "forecast",
               "short_term_forecast": "forecast", "imputation": "impute",
               "impute": "impute", "anomaly_detection": "anomaly",
               "anomaly": "anomaly", "classification": "classify", "classify": "classify"}
    if name not in aliases:
        raise ValueError("Unsupported task_name: %s" % name)
    return aliases[name]


def _float_array(value, name, ndim):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != ndim or not all(value.shape) or not np.isfinite(value).all():
        raise ValueError("%s must be a nonempty finite %d-D numeric array." % (name, ndim))
    return value


def _check_channels(values, config):
    expected = config.get("enc_in")
    if expected is not None and int(expected) != values.shape[-1]:
        raise ValueError("enc_in=%s but input has %d channels." % (expected, values.shape[-1]))


def _standardize(train, values, enabled):
    if not enabled:
        return np.asarray(values, dtype=np.float32)
    axes = tuple(range(train.ndim - 1))
    mean = train.mean(axis=axes, dtype=np.float64)
    std = train.std(axis=axes, dtype=np.float64)
    std = np.where(std < 1e-8, 1.0, std)
    return ((values - mean) / std).astype(np.float32)


def _selected_indices(size, fraction, seed):
    count = max(1, int(size * float(fraction)))
    return np.random.default_rng(int(seed)).permutation(size)[:count].tolist()


def _fitting_values(train, config, task, fraction, seed):
    """For few-shot runs, fit scaling only on the selected training samples."""
    if fraction is None or fraction == 1:
        return train
    if task == "classify":
        return train[_selected_indices(len(train), fraction, seed)]
    windows = WindowDataset(train, config["seq_len"], task, config.get("pred_len", 0), config.get("window_stride", 1))
    indices = _selected_indices(len(windows), fraction, seed)
    covered = np.zeros(len(train), dtype=bool)
    width = int(config["seq_len"]) + (int(config["pred_len"]) if task == "forecast" else 0)
    for index in indices:
        start = windows.starts[index]
        covered[start:start + width] = True
    return train[covered]


class WindowDataset(Dataset):
    def __init__(self, values, seq_len, task, pred_len=0, stride=1, labels=None):
        self.values = np.ascontiguousarray(values, dtype=np.float32)
        self.seq_len, self.pred_len, self.task = int(seq_len), int(pred_len), task
        self.point_labels = labels
        if self.seq_len < 1 or int(stride) < 1 or (task == "forecast" and self.pred_len < 1):
            raise ValueError("seq_len, window_stride and forecast pred_len must be positive.")
        length = self.seq_len + (self.pred_len if task == "forecast" else 0)
        last = len(values) - length
        if last < 0:
            raise ValueError("Split has %d rows; a window needs %d rows." % (len(values), length))
        self.starts = list(range(0, last + 1, int(stride)))
        if self.starts[-1] != last:
            self.starts.append(last)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        x = torch.from_numpy(self.values[start:start + self.seq_len].copy())
        if self.task == "forecast":
            y = torch.from_numpy(self.values[start + self.seq_len:start + self.seq_len + self.pred_len].copy())
        else:
            y = x.clone()
        result = {"x": x, "y": y, "start": start}
        if self.point_labels is not None:
            result["labels"] = torch.from_numpy(self.point_labels[start:start + self.seq_len].copy())
        return result


class ClassificationDataset(Dataset):
    def __init__(self, x, y):
        self.x = np.ascontiguousarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"x": torch.from_numpy(self.x[index].copy()), "y": torch.tensor(self.y[index], dtype=torch.long)}


def _csv_dataset(path, config, flag, task, fraction, seed):
    if task not in {"forecast", "impute"}:
        raise ValueError("CSV supports forecasting and imputation. Use the documented NPZ schema for other tasks.")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV input needs a header row.")
        features = config.get("features", "M")
        if features not in {"M", "S"}:
            raise ValueError("CSV features must be M or S; MS target-only forecasting is not implemented.")
        if features == "S":
            columns = [config.get("target", "OT")]
        else:
            columns = config.get("columns") or [c for c in reader.fieldnames if c != config.get("date_column", "date")]
        if not columns or isinstance(columns, str) or any(c not in reader.fieldnames for c in columns):
            raise ValueError("columns/target must identify existing numeric CSV columns.")
        try:
            values = np.asarray([[float(row[c]) for c in columns] for row in reader], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("CSV columns must be numeric and complete; set columns/date_column explicitly.") from exc
    values = _float_array(values, "CSV values", 2)
    _check_channels(values, config)
    borders = config.get("split_borders")
    if borders is None:
        ratios = config.get("split_ratio", [0.7, 0.1, 0.2])
        if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
            raise ValueError("split_ratio must contain three positive values summing to 1.")
        borders = [int(len(values) * ratios[0]), int(len(values) * (ratios[0] + ratios[1]))]
    if len(borders) != 2 or any(int(v) != v for v in borders):
        raise ValueError("split_borders must contain two integer row offsets [train_end, val_end].")
    train_end, val_end = map(int, borders)
    if not 0 < train_end < val_end < len(values):
        raise ValueError("split_borders must satisfy 0 < train_end < val_end < row_count.")
    fitting = _fitting_values(values[:train_end], config, task, fraction, seed)
    values = _standardize(fitting, values, config.get("scale", True))
    start, end = {"train": (0, train_end), "val": (train_end, val_end), "test": (val_end, len(values))}[flag]
    seq_len = int(config["seq_len"])
    # History may precede the split; all forecast targets stay inside their split.
    if task == "forecast" and flag != "train":
        if start < seq_len:
            raise ValueError("Not enough historical rows before the %s split." % flag)
        start -= seq_len
    return WindowDataset(values[start:end], seq_len, task, config.get("pred_len", 0), config.get("window_stride", 1))


def _npz_dataset(path, config, flag, task, fraction, seed):
    with np.load(path, allow_pickle=False) as arrays:
        if task == "classify":
            required = {"x_train", "y_train", "x_" + flag, "y_" + flag}
            missing = required.difference(arrays.files)
            if missing:
                raise ValueError("Classification NPZ is missing keys: %s" % sorted(missing))
            train = _float_array(arrays["x_train"], "x_train", 3)
            x = _float_array(arrays["x_" + flag], "x_" + flag, 3)
            if x.shape[1:] != train.shape[1:] or x.shape[1] != int(config["seq_len"]):
                raise ValueError("Classification x arrays must have matching [N, seq_len, channels] shapes.")
            _check_channels(x, config)
            classes = int(config.get("num_class", 0))
            for split in {"train", flag}:
                labels = np.asarray(arrays["y_" + split])
                count = len(arrays["x_" + split])
                if labels.ndim != 1 or len(labels) != count or not np.issubdtype(labels.dtype, np.integer):
                    raise ValueError("Classification y arrays must contain one integer class index per sample.")
                if classes < 2 or labels.min() < 0 or labels.max() >= classes:
                    raise ValueError("Class labels must be zero-based integers below configured num_class (>=2).")
            y = np.asarray(arrays["y_" + flag], dtype=np.int64)
            fitting = _fitting_values(train, config, task, fraction, seed)
            return ClassificationDataset(_standardize(fitting, x, config.get("scale", True)), y)
        if task == "anomaly":
            if "train" not in arrays or flag not in arrays:
                raise ValueError("Anomaly NPZ needs train, val and test arrays with shape [time, channels].")
            train = _float_array(arrays["train"], "train", 2)
            x = _float_array(arrays[flag], flag, 2)
            if train.shape[-1] != x.shape[-1]:
                raise ValueError("Anomaly splits must have matching channels.")
            _check_channels(x, config)
            labels = None
            key = flag + "_labels"
            if key in arrays:
                labels = np.asarray(arrays[key])
                if labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all():
                    raise ValueError("%s must contain one binary label per time point." % key)
                labels = labels.astype(np.int64)
            elif flag == "test":
                raise ValueError("Anomaly test evaluation requires test_labels [time] with binary 0/1 values.")
            stride = int(config.get("window_stride", 1))
            if stride > int(config["seq_len"]):
                raise ValueError("Anomaly window_stride must not exceed seq_len, to avoid uncovered points.")
            fitting = _fitting_values(train, config, task, fraction, seed)
            return WindowDataset(_standardize(fitting, x, config.get("scale", True)), config["seq_len"], task, stride=stride, labels=labels)
    raise ValueError("NPZ supports classification and anomaly detection. Use CSV for forecast/imputation.")


def load_dataset(config, flag, subsample_pct=None, seed=0):
    if flag not in {"train", "val", "test"}:
        raise ValueError("flag must be train, val or test.")
    task = canonical_task(config["task_name"])
    if subsample_pct is not None and not 0 < float(subsample_pct) <= 1:
        raise ValueError("subsample_pct is a fraction in (0, 1], e.g. 0.2.")
    data_path = config.get("data_path")
    if not data_path:
        raise ValueError("Set data_path to a CSV (forecast/imputation) or NPZ (classification/anomaly). UEA .ts and raw MSL folders require conversion to the documented NPZ schema.")
    path = Path(config.get("root_path", ".")) / data_path
    if not path.is_file():
        raise FileNotFoundError("Dataset file does not exist: %s" % path)
    if path.suffix.lower() == ".csv":
        dataset = _csv_dataset(path, config, flag, task, subsample_pct, seed)
    elif path.suffix.lower() == ".npz":
        dataset = _npz_dataset(path, config, flag, task, subsample_pct, seed)
    else:
        raise ValueError("Unsupported format: %s. Supply CSV or NPZ; raw benchmark archives require conversion." % path.suffix)
    if flag == "train" and subsample_pct is not None:
        indices = _selected_indices(len(dataset), subsample_pct, seed)
        dataset = Subset(dataset, indices)
    return dataset
