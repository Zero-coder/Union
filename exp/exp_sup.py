import importlib
import os
from typing import Dict

import torch
from torch import nn, optim
import yaml

from data_provider.data_factory import data_provider
from utils.dataloader import BalancedDataLoaderIterator


def read_task_data_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    return config.get("task_dataset", {})


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []
    for task_name, task_config in task_data_config.items():
        task_config["max_batch"] = default_batch_size
        task_data_config_list.append([task_name, task_config])
    return task_data_config_list


class Exp_All_Task:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task_data_config = read_task_data_config(args.task_data_config_path)
        self.task_data_config_list = get_task_data_config_list(self.task_data_config, args.batch_size)
        self.model = self._build_model()

    def _build_model(self):
        module = importlib.import_module("models." + self.args.model)
        return module.Model(self.args, self.task_data_config_list).to(self.device)

    def _get_data(self, flag):
        loaders = []
        for _, task_config in self.task_data_config.items():
            _, loader = data_provider(self.args, task_config, flag, ddp=False)
            loaders.append(loader)
        return BalancedDataLoaderIterator(loaders)

    def train(self, setting):
        os.makedirs(os.path.join(self.args.checkpoints, setting), exist_ok=True)
        loader = self._get_data("train")
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = nn.MSELoss()
        self.model.train()
        for epoch in range(self.args.train_epochs):
            losses = []
            for (x, y), task_id in loader:
                x = x.to(self.device)
                task_name = self.task_data_config_list[task_id][1]["task_name"]
                outputs = self.model(x_enc=x, task_id=task_id, task_name=task_name)
                if "logits" in outputs:
                    loss = nn.CrossEntropyLoss()(outputs["logits"], y.to(self.device))
                elif "forecast" in outputs:
                    pred = outputs["forecast"][:, : y.size(1), : y.size(2)]
                    loss = criterion(pred, y.to(self.device))
                else:
                    pred = outputs["reconstruction"][:, : y.size(1), : y.size(2)]
                    loss = criterion(pred, y.to(self.device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                optimizer.step()
                losses.append(loss.item())
            print(f"Epoch {epoch + 1}: train_loss={sum(losses) / max(len(losses), 1):.6f}")
        return self.model

    def test(self, setting, load_pretrain=False):
        loader = self._get_data("test")
        self.model.eval()
        with torch.no_grad():
            for (x, _), task_id in loader:
                task_name = self.task_data_config_list[task_id][1]["task_name"]
                outputs = self.model(x_enc=x.to(self.device), task_id=task_id, task_name=task_name)
                print(f"test task_id={task_id} outputs={list(outputs.keys())}")
                break
