# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR

import os
from typing import List, Any, Dict, Optional, Tuple, Callable, Union
from tqdm import tqdm
import datetime
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_
from collections import abc


# 未来会添加的功能: 多GPU, 混合精度训练. 断点续训
#

__all__ = ["LModule", "LDataModule", "Trainer"]
# 这里的batch_idx[+1], epoch_idx[+0]


class LModule:
    def __init__(self, model: Module, optim: Optimizer,
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        self.model = model
        self.optim = optim
        self.hparams = hparams
        #

        self.mes = {}  # type: Dict[str, float]
        self.prog_bar_mean = {}
        #

    def log(self, k: str, v: Union[Tensor, float], *, prog_bar_mean=True):
        """
        prog_bar_mean: 在prog_bar中显示的是整个epoch的均值. (一般loss, acc用均值. lr不用均值)
        """
        # 如何log. 我们调用lmodule的log, 将信息存储在lmodule中
        # 当单次迭代结束时, 会修改lmodule._mes, 随后加到trainer的全局log中...
        if isinstance(v, Tensor):
            v = v.item()
        self.mes[k] = v
        self.prog_bar_mean[k] = prog_bar_mean

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_from_checkpoint(self, ckpt_path: str) -> None:
        self.model.load_state_dict(torch.load(ckpt_path))

    def save_checkpoint(self, ckpt_path: str) -> None:
        torch.save(self.model.state_dict(), ckpt_path)

    def epoch_end(self) -> None:
        # fit. 用于lr_schedules的处理
        # 这里log的信息不会出现在prog_bar中(但会在tensorboard中).
        # log要在lrs.step之前
        pass

    def _batch_to_device(self, batch: Any, device: Device) -> Any:
        # tree的深搜. 对python object(int, float)报错
        #   处理list, tuple, dict, Tensor
        if isinstance(batch, Tensor):
            return batch.to(device)
        #
        if isinstance(batch, abc.Sequence):
            res = []
            for b in batch:
                res.append(self._batch_to_device(b, device))
        elif isinstance(batch, abc.Mapping):
            res = {}
            for k, v in batch.items():
                res[k] = self._batch_to_device(v, device)
        else:
            raise TypeError(f"batch: {batch}, {type(batch)}")
        return res

    def batch_to_device(self, batch: Any, device: Device) -> Any:
        # fit/test.
        return self._batch_to_device(batch, device)

    def optimizer_step(self) -> None:
        # fit. 用于optim, lr_schedules的处理.
        # log要在lrs.step之前. optim.step要在lrs.step之前.
        # 已过optim.zero_grad, loss.backward
        self.optim.step()

    def training_step(self, batch: Any) -> Tensor:
        # fit
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        #   此设计用于: GAN
        raise NotImplementedError

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # fit. no_grad环境
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        raise NotImplementedError

    def test_step(self, batch: Any) -> None:
        # test. no_grad环境
        raise NotImplementedError


# if __name__ == "__main__":
#     x = {"tensor": torch.tensor([1, 2]), 0: [
#         torch.tensor([1, 2]), (torch.tensor([1]),)]}
#     print(LModule(None, None)._batch_to_device(x, Device('cuda')))
#     try:
#         x[0].append(1)
#         print(LModule(None, None)._batch_to_device(x, Device('cuda')))
#     except TypeError as e:
#         print(e)
#     exit(0)


class LDataModule:
    # @staticmethod
    # def default_collate_fn(batch: List[Any]) -> Tuple[Tensor]:
    #     # batch: e.g. List[dataset[0], dataset[1]...]
    #     res = []
    #     for x in zip(*batch):
    #         if isinstance(x[0], Tensor):
    #             res.append(torch.stack(x))  # e.g. data
    #         else:
    #             res.append(torch.tensor(x))  # e.g. labels
    #     return tuple(res)

    def __init__(self, train_dataset: Optional[Dataset], val_dataset: Optional[Dataset], test_dataset: Optional[Dataset],
                 batch_size_train: int, num_workers: int = 0,
                 collate_fn: Optional[Callable[[List[Any]], Any]] = None, *,
                 shuffle_train: bool = True, pin_memory_train: bool = True) -> None:

        self.train_dataloader = None  # type: DataLoader
        self.val_dataloader = None  # type: DataLoader
        self.test_dataloader = None  # type: DataLoader

        batch_size_test = batch_size_train * 2
        # collate_fn = collate_fn if collate_fn is not None else self.default_collate_fn
        #
        if train_dataset:
            self.train_dataloader = DataLoader(train_dataset, batch_size_train, shuffle=shuffle_train,
                                               num_workers=num_workers, pin_memory=pin_memory_train,
                                               drop_last=True, collate_fn=collate_fn)
        if val_dataset:
            self.val_dataloader = DataLoader(val_dataset, batch_size_test, shuffle=False,
                                             num_workers=num_workers, pin_memory=False,
                                             drop_last=False, collate_fn=collate_fn)
        if test_dataset:
            self.test_dataloader = DataLoader(test_dataset, batch_size_test, shuffle=False,
                                              num_workers=num_workers, pin_memory=False,
                                              drop_last=False, collate_fn=collate_fn)


class Trainer:
    def __init__(self, lmodel: LModule, gpus: bool, max_epochs: int, runs_dir: str, *,
                 log_every_n_steps: int = 5, gradient_clip_norm: float = None, benchmark: bool = None) -> None:
        # 现在只支持单gpu, 多gpu以后再扩展
        self.lmodel = lmodel
        self.device = Device("cuda") if gpus else Device("cpu")
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.gradient_clip_norm = gradient_clip_norm

        #
        time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S.%f")
        self.ckpt_dir = os.path.join(runs_dir, time, "checkpoints")
        self.tb_dir = os.path.join(
            runs_dir, time, "runs")  # tensorboard
        self.hparams_path = os.path.join(
            runs_dir, time, "hparams.yaml")
        self.result_path = os.path.join(
            runs_dir, time, "result.yaml")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        #
        benchmark = benchmark if benchmark is not None else True
        torch.backends.cudnn.benchmark = False \
            if torch.backends.cudnn.deterministic else benchmark
        #
        self.logger = SummaryWriter(self.tb_dir)
        self.best_metrics = -1e10  # model save
        self.best_ckpt_path = None  # type: str
        self.last_ckpt_path = None  # type: str
        self.global_step = 0
        self.global_epoch = -1
        #
        hparams = self.lmodel.hparams
        self.save_hparams(hparams if hparams is not None else {})

    def check_hparams(self, hparams: Any) -> Any:
        # 只支持List, Dict, int, float, str
        # tuple -> list
        # 其他的不存储. 例如: collate_fn
        ###
        # 树的深搜
        if isinstance(hparams, (int, float, str)):  # bool是int的子类
            return hparams
        if isinstance(hparams, abc.Sequence):
            res = []
            for hp in hparams:
                res.append(self.check_hparams(hp))
        elif isinstance(hparams, abc.Mapping):
            res = {}
            for k, v in hparams.items():
                res[k] = self.check_hparams(v)
        else:
            res = "!!ignored"
        return res

    def save_hparams(self, hparams: Dict[str, Any]) -> None:
        with open(self.hparams_path, "w") as f:
            saved_hparams = self.check_hparams(hparams)
            yaml.dump(saved_hparams, f)

    @staticmethod
    def _sum_to_mean(log_mes: Dict[str, float], n: int, inplace: bool = False) -> Dict[str, float]:
        if not inplace:
            log_mes = log_mes.copy()
        for k, v in log_mes.items():
            log_mes[k] = v / n
        return log_mes

    def _logger_add_scalars(self, mes: Dict[str, float], step: int) -> None:
        # mes(not sum)
        for k, v in mes.items():
            self.logger.add_scalar(k, v, global_step=step)

    def _remove_ckpt(self, mode: str) -> None:
        if mode == "best" and self.best_ckpt_path is not None:
            os.remove(self.best_ckpt_path)
        elif mode == "last" and self.last_ckpt_path is not None:
            os.remove(self.last_ckpt_path)

    def _epoch_end(self, mes: Dict[str, float], metric: float) -> None:
        # 1. 模型保存
        ckpt_dir = self.ckpt_dir
        if metric > self.best_metrics:
            # 保存
            self._remove_ckpt("best")
            self.best_metrics = metric
            ckpt_fname = f"best-epoch={self.global_epoch}-metrics={metric}.ckpt"
            self.best_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
            self.lmodel.save_checkpoint(self.best_ckpt_path)
            print(f"- best model, saving model `{ckpt_fname}`")
        #
        self._remove_ckpt("last")
        ckpt_fname = f"last-epoch={self.global_epoch}-metrics={metric}.ckpt"
        self.last_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        self.lmodel.save_checkpoint(self.last_ckpt_path)
        # 2. 结果保存
        with open(self.result_path, "a") as f:
            yaml.dump({f"Epoch={self.global_epoch}": mes}, f)

    def _add_new_mes(self, mes: Dict[str, float], new_mes: Dict[str, float]) -> None:
        for k, v in new_mes.items():
            if k not in mes:
                mes[k] = 0
            mes[k] += v

    @staticmethod
    def _get_log_mes(mean_mes: Dict[str, float], new_mes: Dict[str, float], prog_bar_mean: Dict[str, bool]):
        res = {}
        # 假设mes, new_mes, prog_bar_mean的k相同
        keys = mean_mes.keys()
        for k in keys:
            if prog_bar_mean[k] is True:
                res[k] = mean_mes[k]
            else:
                res[k] = new_mes[k]
        return res

    def _train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        lmodel = self.lmodel
        model = lmodel.model
        model.train()
        model.to(self.device)
        #
        mes = {}
        #
        with tqdm(total=len(dataloader),
                  desc=f"Epoch {self.global_epoch}") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.global_step += 1
                lmodel.mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                loss = lmodel.training_step(batch)
                if loss is not None:
                    lmodel.optim.zero_grad()
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        clip_grad_norm_(
                            model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=True)
                    lmodel.optimizer_step()
                #
                new_mes = lmodel.mes
                self._add_new_mes(mes, new_mes)
                # tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(new_mes, self.global_step)
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, new_mes, lmodel.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update()
            self._sum_to_mean(mes, len(dataloader), inplace=True)

        return mes

    def _train(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        lmodel = self.lmodel
        for _ in range(self.global_epoch + 1, self.max_epochs):
            self.global_epoch += 1
            mes = self._train_epoch(train_dataloader)
            #
            metrics, val_mes = self._val(val_dataloader)
            mes.update(val_mes)
            # 后处理
            lmodel.mes.clear()
            lmodel.epoch_end()
            new_mes = lmodel.mes
            self._logger_add_scalars(new_mes, self.global_epoch)
            mes.update(new_mes)
            #
            self._epoch_end(mes, metrics)

    @torch.no_grad()
    def _val(self, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        lmodel = self.lmodel
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        metrics = 0.  # sum stat

        #
        mes = {}
        with tqdm(total=len(dataloader), desc="  Val: ") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                lmodel.mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                _m = lmodel.validation_step(batch)
                metrics += _m.item() if isinstance(_m, Tensor) else _m
                new_mes = lmodel.mes
                self._add_new_mes(mes, new_mes)
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, new_mes, lmodel.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update()
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        self._logger_add_scalars(mes, self.global_epoch)
        return metrics / len(dataloader), mes

    @ torch.no_grad()
    def _test(self, dataloader: DataLoader) -> Dict[str, float]:
        lmodel = self.lmodel
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        #
        mes = {}  # sum stat
        with tqdm(total=len(dataloader), desc="Test: ") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                lmodel.mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                lmodel.test_step(batch)
                new_mes = self.lmodel.mes
                self._add_new_mes(mes, new_mes)  # LModule.log的内容
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, new_mes, lmodel.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update()
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        self._logger_add_scalars(mes, self.global_epoch)
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
    # 更多的examples见 `https://github.com/Jintao-Huang/ml_alg/blob/main/tutorials/pl`
    import torch.nn as nn
    import torch.optim as optim
    try:
        from . import MLP_L2, XORDataset, accuracy_score
    except ImportError:
        from models import MLP_L2
        from datasets import XORDataset
        from metrics import accuracy_score
    #

    train_dataset = XORDataset(512)
    val_dataset = XORDataset(256)
    test_dataset = XORDataset(256)
    ldm = LDataModule(train_dataset, val_dataset, test_dataset, 64)

    #
    model = MLP_L2(2, 4, 1)
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)

    class MyLModule(LModule):
        def __init__(self, model: Module, optim: Optimizer) -> None:
            super(MyLModule, self).__init__(model, optim, {"model": "MLP_2"})
            self.loss_fn = nn.BCEWithLogitsLoss()
            self.lrs = MultiStepLR(optim, [10, 50], 0.1)

        def epoch_end(self) -> None:
            # fit. 用于lr_schedules的处理
            self.log("lr0", self.lrs.get_last_lr()[0])
            self.lrs.step()

        def training_step(self, batch: Any) -> Tensor:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            loss = self.loss_fn(y, y_batch.float())  # type: Tensor
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch: Any) -> Union[Tensor, float]:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            y = y >= 0
            acc = accuracy_score(y, y_batch)
            self.log("val_acc", acc)
            return acc

        def test_step(self, batch: Any) -> None:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            y = y >= 0
            acc = accuracy_score(y, y_batch)
            self.log("test_acc", acc)
    #
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    RUNS_DIR = os.path.join(_ROOT_DIR, "runs")
    os.makedirs(RUNS_DIR, exist_ok=True)
    #
    runs_dir = os.path.join(RUNS_DIR, "test_mini_pl")
    lmodel = MyLModule(model, optimizer)
    #
    trainer = Trainer(lmodel, True, 100, runs_dir)
    trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
    trainer.test(ldm.test_dataloader)
    del lmodel.model
    lmodel.model = MLP_L2(2, 4, 1)
    lmodel.load_from_checkpoint(trainer.best_ckpt_path)
    #
    trainer.test(ldm.test_dataloader)
