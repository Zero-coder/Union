import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from union import UnionConfig, UnionModel
from union.data import SyntheticConfig, SyntheticTimeSeries


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_loss(task: str):
    if task == "classify":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def train(config: Dict[str, Any]) -> None:
    torch.manual_seed(int(config.get("seed", 42)))
    task = config["task"]
    model_config = UnionConfig(
        seq_len=config["seq_len"],
        pred_len=config["pred_len"],
        num_vars=config["num_vars"],
        num_classes=config.get("num_classes", 0),
        patch_len=config.get("patch_len", 16),
        stride=config.get("stride", 8),
        d_model=config.get("d_model", 128),
        depth=config.get("depth", 3),
        n_heads=config.get("n_heads", 4),
        num_experts=config.get("num_experts", 4),
        top_k=config.get("top_k", 2),
        dropout=config.get("dropout", 0.1),
    )
    dataset = SyntheticTimeSeries(
        SyntheticConfig(
            task=task,
            seq_len=model_config.seq_len,
            pred_len=model_config.pred_len,
            num_vars=model_config.num_vars,
            num_classes=model_config.num_classes,
        )
    )
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 16), shuffle=True)
    model = UnionModel(model_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate", 3e-4))
    criterion = build_loss(task)

    model.train()
    steps = int(config.get("train_steps", 20))
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)

        outputs = model(x, task=task)
        if task == "forecast":
            pred = outputs["forecast"]
            loss = criterion(pred, y)
        elif task in {"impute", "anomaly"}:
            pred = outputs["reconstruction"]
            loss = criterion(pred, y)
        else:
            loss = criterion(outputs["logits"], y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step == steps or step % 5 == 0:
            print(f"step={step:04d} task={task} loss={loss.item():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or smoke-test Union.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
