from torch import optim

from exp.exp_sup import Exp_All_Task as SupervisedExp
from utils.losses import UnifiedMaskRecLoss


class Exp_All_Task(SupervisedExp):
    def train(self, setting):
        loader = self._get_data("train")
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        criterion = UnifiedMaskRecLoss()
        self.model.train()
        for epoch in range(self.args.train_epochs):
            losses = []
            for (x, _), task_id in loader:
                x = x.to(self.device)
                task_name = self.task_data_config_list[task_id][1]["task_name"]
                outputs = self.model(x_enc=x, task_id=task_id, task_name=task_name, enable_mask=True)
                loss = criterion(outputs, x)["loss"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(f"Pretrain epoch {epoch + 1}: train_loss={sum(losses) / max(len(losses), 1):.6f}")
        return self.model
