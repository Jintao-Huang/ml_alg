import torch
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer

import os
from typing import List, Any, Dict, Optional, Tuple
from tqdm import tqdm
import datetime
import yaml

__all__ = ["LModule", "Trainer"]
# 可以再加: lr_scheduler


class LModule:
    def __init__(self, model: Module, optim: Optimizer, default_root_dir: str,
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.optim = optim
        self.default_root_dir = default_root_dir
        #
        time = datetime.datetime.now().strftime("%Y:%m:%d:%H:%M:%S.%f")
        self.ckpt_dir = os.path.join(default_root_dir, time, "checkpoints")
        self.tb_dir = os.path.join(
            default_root_dir, time, "runs")  # tensorboard
        self.hparams_path = os.path.join(
            default_root_dir, time, "hparams.yaml")
        self.result_path = os.path.join(
            default_root_dir, time, "result.yaml")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.mes = {}  # type: Dict[str, float]
        #
        self.save_hparams(hparams if hparams is not None else {})

    def save_hparams(self, hparams: Dict[str, Any]) -> None:
        with open(self.hparams_path, "w") as f:
            yaml.dump(hparams, f)

    def log(self, k: str, v: float):
        # 如何log. 我们调用lmodule的log, 将信息存储在lmodule中
        # 当单次迭代结束时, 会修改lmodule._mes, 随后加到trainer的全局log中...
        self.mes[k] = float(v)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_from_checkpoint(self, ckpt_fname: str) -> None:
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
        self.model.load_state_dict(torch.load(ckpt_path))

    def save_checkpoint(self, ckpt_fname: str) -> None:
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
        torch.save(self.model.state_dict(), ckpt_path)

    #
    def optimizer_step(self) -> None:
        ...

    def batch_to_device(self, batch: Any, device: Device) -> Any:
        ...

    def training_step(self, batch: Any) -> Tensor:
        # 返回的Tensor(loss)用于优化
        ...

    def validation_step(self, batch: Any) -> float:
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        ...

    def test_step(self, batch: Any) -> None:
        ...


class Trainer:
    def __init__(self, lmodel: LModule, gpus: bool, max_epochs: int,
                 log_every_n_steps: int = 5) -> None:
        # 现在只支持单gpu和cpu, 多gpu以后再扩展
        self.lmodel = lmodel
        self.device = Device("cuda") if gpus else Device("cpu")
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        #
        self.logger = SummaryWriter(self.lmodel.tb_dir)
        self.best_metrics = -1e10  # model save
        self.best_ckpt_path = None
        self.last_ckpt_path = None

    @staticmethod
    def _sum_to_mean(log_mes: Dict[str, float], n: int, inplace: bool = False) -> Dict[str, float]:
        if not inplace:
            log_mes = log_mes.copy()
        for k, v in log_mes.items():
            log_mes[k] = v / n
        return log_mes

    def _get_log_string(self, log_mes: Dict[str, float]) -> str:
        # log_mes(not sum)
        log_string = ""
        for k, v in log_mes.items():
            log_string += f"{k}={v:.4f} "
        return log_string

    def _logger_add_scalars(self, mes: Dict[str, float], step: int) -> None:
        # mes(not sum)
        for k, v in mes.items():
            self.logger.add_scalar(k, v, global_step=step)

    def _remove_ckpt(self, mode: str) -> None:
        if mode == "best" and self.best_ckpt_path is not None:
            os.remove(self.best_ckpt_path)
        elif mode == "last" and self.last_ckpt_path is not None:
            os.remove(self.last_ckpt_path)

    def _epoch_end(self, epoch_idx: int, mes: Dict[str, float], metric: float) -> None:
        # n = epoch_idx+1
        # 1. 模型保存
        ckpt_dir = self.lmodel.ckpt_dir
        if metric > self.best_metrics:
            # 保存
            self._remove_ckpt("best")
            self.best_metrics = metric
            ckpt_fname = f"best-epoch={epoch_idx +1}-metrics={metric}.ckpt"
            self.best_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
            self.lmodel.save_checkpoint(ckpt_fname)
        #
        self._remove_ckpt("last")
        ckpt_fname = f"last-epoch={epoch_idx + 1}-metrics={metric}.ckpt"
        self.last_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        self.lmodel.save_checkpoint(ckpt_fname)
        # 2. 结果保存
        with open(self.lmodel.result_path, "a") as f:
            yaml.dump({f"Epoch={epoch_idx + 1}": mes}, f)

    def _add_new_mes(self, mes: Dict[str, float], new_mes: Dict[str, float]) -> None:
        for k, v in new_mes.items():
            if k not in mes:
                mes[k] = 0
            mes[k] += v

    def _train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        lmodel = self.lmodel
        model = lmodel.model
        optim = lmodel.optim
        global_step = 0
        #
        for epoch_idx in range(self.max_epochs):
            model.train()
            model.to(self.device)
            #
            mes = {}
            #
            prog_bar = tqdm(enumerate(train_dataloader),
                            total=len(train_dataloader))
            for batch_idx, batch in prog_bar:
                global_step += 1
                batch = lmodel.batch_to_device(batch, self.device)
                loss = lmodel.training_step(batch)
                new_mes = self.lmodel.mes
                self._add_new_mes(mes, new_mes)
                # tensorboard
                if global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(new_mes, global_step)
                new_mes.clear()
                #
                optim.zero_grad()
                loss.backward()
                lmodel.optimizer_step()
                log_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_string = f"Epoch {epoch_idx}: " + \
                    self._get_log_string(log_mes)
                prog_bar.set_description(log_string)

            self._sum_to_mean(mes, len(train_dataloader), inplace=True)
            #
            metrics = self._val(val_dataloader, mes, epoch_idx)
            self._epoch_end(epoch_idx, mes, metrics)

    @torch.no_grad()
    def _val(self, dataloader: DataLoader, mes: Dict[str, float], epoch_idx: int) -> float:
        lmodel = self.lmodel
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        metrics = 0.  # sum stat

        #
        val_mes = {}
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in prog_bar:
            batch = lmodel.batch_to_device(batch, self.device)
            metrics += float(lmodel.validation_step(batch))
            new_mes = self.lmodel.mes
            self._add_new_mes(val_mes, new_mes)
            new_mes.clear()
            log_mes = self._sum_to_mean(val_mes, batch_idx + 1)
            log_string = "    Val: " + self._get_log_string(log_mes)
            prog_bar.set_description(log_string)
        self._sum_to_mean(val_mes, len(dataloader), inplace=True)
        self._logger_add_scalars(val_mes, epoch_idx + 1)
        mes.update(val_mes)
        return metrics / len(dataloader)

    @torch.no_grad()
    def _test(self, dataloader: DataLoader) -> Dict[str, float]:
        lmodel = self.lmodel
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        #
        mes = {}  # sum stat
        prog_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, batch in prog_bar:
            batch = lmodel.batch_to_device(batch, self.device)
            lmodel.test_step(batch)
            new_mes = self.lmodel.mes
            self._add_new_mes(mes, new_mes)  # LModule.log的内容
            new_mes.clear()
            log_mes = self._sum_to_mean(mes, batch_idx + 1)
            log_string = self._get_log_string(log_mes)
            prog_bar.set_description(log_string)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        return mes

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        # 主要内容: 记录训练的过程, 包括train每个迭代的信息; val每个epoch的信息
        device_r = next(self.lmodel.model.parameters()).device
        self._train(train_dataloader, val_dataloader)
        self.lmodel.model.to(device_r)

    def test(self, dataloader: DataLoader) -> Dict[str, float]:
        # 主要内容: 不需要记录信息, 只需要打印测试结果即可.
        device_r = next(self.lmodel.model.parameters()).device
        mes = self._test(dataloader)
        self.lmodel.model.to(device_r)
        return mes


if __name__ == "__main__":
    import pytorch_lightning as pl
    import torch.nn as nn
    import torch.optim as optim
    try:
        from . import MLP_L2, XORDataset, RUNS_DIR, accuracy
    except ImportError:
        from models import MLP_L2
        from datasets import XORDataset
        from config import RUNS_DIR
        from metrics import accuracy
    #
    train_dataset = XORDataset(512)
    train_dataloader = DataLoader(train_dataset, 64, True)
    val_dataset = XORDataset(256)
    val_dataloader = DataLoader(val_dataset, 64, True)
    test_dataset = XORDataset(256)
    test_dataloader = DataLoader(test_dataset, 128, False)
    #
    model = MLP_L2(2, 4, 1)
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)

    class MyLModule(LModule):
        def __init__(self, model: Module, optim: Optimizer, default_root_dir: str) -> None:
            super(MyLModule, self).__init__(model, optim,
                                            default_root_dir, {"model": "MLP_2"})
            self.loss_fn = nn.BCEWithLogitsLoss()

        def optimizer_step(self) -> None:
            self.optim.step()

        def batch_to_device(self, batch: Any, device: Device) -> Any:
            x_batch, y_batch = batch
            return x_batch.to(device), y_batch.to(device)

        def training_step(self, batch: Any) -> Tensor:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            loss = self.loss_fn(y, y_batch.float())  # type: Tensor
            self.log("train_loss", loss.item())
            return loss

        def validation_step(self, batch: Any) -> float:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            acc = accuracy(y, y_batch)
            self.log("val_acc", acc)
            return acc

        def test_step(self, batch: Any) -> None:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            acc = accuracy(y, y_batch)
            self.log("test_acc", acc)

    default_root_dir = os.path.join(RUNS_DIR, "test_mini_pl")
    lmodel = MyLModule(model, optimizer, default_root_dir)
    #
    trainer = Trainer(lmodel, True, 100)
    trainer.fit(train_dataloader, val_dataloader)
    trainer.test(test_dataloader)
    model_name = "m.ckpt"
    lmodel.save_checkpoint(model_name)
    del lmodel.model
    lmodel.model = MLP_L2(2, 4, 1)
    lmodel.load_from_checkpoint(model_name)
    #
    trainer.test(test_dataloader)
