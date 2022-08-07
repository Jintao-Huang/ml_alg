# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
try:
    from .cv import print_model_info
except ImportError:
    pass
import torch
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler as LRScheduler

import os
from typing import List, Any, Dict, Optional, Tuple, Callable, Union, Sequence, Mapping, Literal
from tqdm import tqdm
import datetime
import yaml
from torch.nn.utils.clip_grad import clip_grad_norm_

# 未来会添加的功能: 多GPU, 混合精度训练. 断点续训. 梯度累加
#

__all__ = ["LModule", "LDataModule", "Trainer"]


class LModule:
    def __init__(self, model: Module, optimizer: Optimizer,
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        """hparams: 需要保存的超参数. """
        # 一般: 定义损失函数, 学习率管理器. (优化器, 模型)
        self.model = model
        self.optimizer = optimizer
        self.hparams: Dict[str, Any] = hparams if hparams is not None else {}
        self.trainer: Optional["Trainer"] = None

    @property
    def global_step(self) -> int:
        # global_step是从1开始的
        return self.trainer.global_step if self.trainer is not None else 0

    @property
    def global_epoch(self) -> int:
        # global_epoch是从0开始的
        return self.trainer.global_epoch if self.trainer is not None else -1

    @property
    def device(self) -> Optional[Device]:
        return self.trainer.device if self.trainer is not None else None

    def log(self, k: str, v: Union[Tensor, float], *, prog_bar_mean=True) -> None:
        """
        prog_bar_mean: 在prog_bar中显示的是整个epoch的均值. (一般loss, acc用均值. lr不用均值)
        """
        # 如何log. 我们调用lmodule的log, 将信息存储在lmodule中
        # 当单次迭代结束时, 会修改lmodule._mes, 随后加到trainer的全局log中...
        if self.trainer is None:
            raise ValueError(f"self.trainer: {self.trainer}")
        if isinstance(v, Tensor):
            v = v.item()
        self.trainer.new_mes[k] = v
        self.trainer.prog_bar_mean[k] = prog_bar_mean

    def log_dict(self, _dict: Dict[str, Union[Tensor, float]], *, prog_bar_mean=True) -> None:
        # 参数见self.log
        for k, v in _dict.items():
            self.log(k, v, prog_bar_mean=prog_bar_mean)

    def __call__(self, *args, **kwargs) -> Any:
        return self.model(*args, **kwargs)

    def load_from_checkpoint(self, ckpt_path: str) -> None:
        device = next(self.model.parameters()).device
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))

    def save_checkpoint(self, ckpt_path: str) -> None:
        torch.save(self.model.state_dict(), ckpt_path)

    def training_epoch_start(self) -> None:
        # [fit]用于模型to(device), 特别是有多个model的情况下会使用(dqn)
        self.model.train()
        self.model.to(self.device)

    def training_epoch_end(self) -> None:
        # [fit]用于lr_schedules的处理
        # 这里log的信息不会出现在prog_bar中(但会在tensorboard中).
        # log要在lrs.step之前
        pass

    def _batch_to_device(self, batch: Any, device: Device) -> Any:
        # tree的深搜. 对python object(int, float)报错
        #   处理list, tuple, dict, Tensor
        if isinstance(batch, Tensor):
            # https://pytorch-lightning.readthedocs.io/en/stable/_modules/pytorch_lightning/utilities/apply_func.html?highlight=non_blocking#
            # 同pytorch-lightning
            non_blocking = False
            if device not in (Device("cpu"), "cpu"):
                non_blocking = True
            return batch.to(device=device, non_blocking=non_blocking)
        #
        if isinstance(batch, Sequence):
            res = []
            for b in batch:
                res.append(self._batch_to_device(b, device))
            if isinstance(batch, tuple):
                res = tuple(res)
        elif isinstance(batch, Mapping):
            res = {}
            for k, v in batch.items():
                res[k] = self._batch_to_device(v, device)
        else:
            raise TypeError(f"batch: {batch}, {type(batch)}")
        return res

    def batch_to_device(self, batch: Any, device: Device) -> Any:
        # [train/val/test]
        return self._batch_to_device(batch, device)

    def optimizer_step(self) -> None:
        # [train]. 用于optimizer, lr_schedules的处理.
        # log要在lrs.step之前. optimizer.step要在lrs.step之前.
        # 已过optimizer.zero_grad, loss.backward
        self.optimizer.step()

    def training_step(self, batch: Any) -> Tensor:
        # [train]
        # 返回的Tensor(loss)用于优化. 如果返回None, 则training_step内进行自定义optimizer_step.
        #   此设计用于: GAN
        raise NotImplementedError

    def validation_step(self, batch: Any) -> Union[Tensor, float]:
        # [val]. no_grad环境
        # 返回的float用于模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数)
        #   若不提供val_dataloader, 则该函数可以不实现
        raise NotImplementedError

    def test_step(self, batch: Any) -> None:
        # [test]. no_grad环境
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

        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None

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
    def __init__(self, lmodel: LModule, device: Union[str, Device], max_epochs: int, runs_dir: str, *,
                 log_every_n_steps: int = 5, prog_bar_n_steps: int = 10,
                 gradient_clip_norm: float = None, benchmark: bool = None) -> None:
        """
        log_every_n_steps: 几步需要将信息log入tensorboard. 使用global_step. 
            这不会修改prog_bar(进度条)的显示(每个step都会更新). 
        prog_bar_n_steps: 进度条的显示的频率. 使用batch_idx
        gradient_clip_norm: 梯度裁剪, 防止梯度爆炸
        benchmark: # https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            Pytorch默认False. 该函数的benchmark行为与Pytorch Lightning行为一致 
            True: 可以加速训练, 但是会造成不可复现. 
        """
        self.lmodel = lmodel
        self.lmodel.trainer = self
        self.device = Device(device) if isinstance(device, str) else device
        self.max_epochs = max_epochs
        self.log_every_n_steps = log_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.gradient_clip_norm = gradient_clip_norm

        #
        time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S.%f")
        runs_dir = os.path.join(runs_dir, time)
        print(f"runs_dir: {runs_dir}")
        self.ckpt_dir = os.path.join(runs_dir, "checkpoints")
        self.tb_dir = os.path.join(
            runs_dir, "runs")  # tensorboard
        self.hparams_path = os.path.join(
            runs_dir, "hparams.yaml")
        self.result_path = os.path.join(
            runs_dir, "result.yaml")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        #
        benchmark = benchmark if benchmark is not None else True
        torch.backends.cudnn.benchmark = False \
            if torch.backends.cudnn.deterministic else benchmark
        #
        self.logger = SummaryWriter(self.tb_dir)
        self.best_metrics = -1e10  # model save
        self.best_ckpt_path: str = None
        self.last_ckpt_path: str = None
        self.global_step = 0
        self.global_epoch = -1
        #
        hparams = self.lmodel.hparams
        self.save_hparams(hparams)
        # 用于log. 含义见LModule.log
        self.new_mes: Dict[str, float] = {}
        self.prog_bar_mean = {}

    def check_hparams(self, hparams: Any) -> Any:
        # 只支持List, Dict, int, float, str
        # tuple -> list
        # 其他的不存储. 例如: collate_fn
        ###
        # 树的深搜
        if isinstance(hparams, (int, float, str)):  # bool是int的子类
            return hparams
        if isinstance(hparams, Sequence):
            res = []
            for hp in hparams:
                res.append(self.check_hparams(hp))
        elif isinstance(hparams, Mapping):
            res = {}
            for k, v in hparams.items():
                res[k] = self.check_hparams(v)
        else:
            res = str(hparams)
        return res

    def save_hparams(self, hparams: Dict[str, Any]) -> None:
        with open(self.hparams_path, "w") as f:
            saved_hparams = self.check_hparams(hparams)
            yaml.dump(saved_hparams, f)

    @staticmethod
    def _sum_to_mean(log_mes: Dict[str, float], n: int, inplace: bool = False) -> Dict[str, float]:
        if not inplace:
            log_mes = log_mes.copy()
        for k in log_mes.keys():
            log_mes[k] /= n
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

    def _epoch_end(self, mes: Dict[str, float], metric: Optional[float]) -> bool:
        # 1. 模型保存
        ckpt_dir = self.ckpt_dir
        is_best = False
        if metric is not None and metric >= self.best_metrics:  # 含等于
            # 保存
            self._remove_ckpt("best")
            self.best_metrics = metric
            ckpt_fname = f"best-epoch={self.global_epoch}-metrics={metric}.ckpt"
            self.best_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
            self.lmodel.save_checkpoint(self.best_ckpt_path)
            print(f"- best model, saving model `{ckpt_fname}`")
            is_best = True
        #
        self._remove_ckpt("last")
        ckpt_fname = f"last-epoch={self.global_epoch}-metrics={metric}.ckpt"
        self.last_ckpt_path = os.path.join(ckpt_dir, ckpt_fname)
        self.lmodel.save_checkpoint(self.last_ckpt_path)
        # 2. 结果保存
        with open(self.result_path, "a") as f:
            yaml.dump({f"Epoch={self.global_epoch}": mes}, f)
        return is_best

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

    def _train_epoch(self, lmodel: LModule, dataloader: DataLoader) -> Dict[str, float]:
        model = lmodel.model
        lmodel.training_epoch_start()
        #
        mes = {}
        #
        with tqdm(total=len(dataloader),
                  desc=f"Epoch {self.global_epoch}") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.global_step += 1
                self.new_mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                loss = lmodel.training_step(batch)
                if loss is not None:
                    # set_to_none可以增加时间/内存性能. 该行为与Pytorch Lightning默认行为不一致
                    lmodel.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if self.gradient_clip_norm is not None:
                        clip_grad_norm_(
                            model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=True)
                    lmodel.optimizer_step()
                #
                self._add_new_mes(mes, self.new_mes)
                # tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(self.new_mes, self.global_step)
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, self.new_mes, self.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                if batch_idx % self.prog_bar_n_steps == 0:
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        # 后处理
        self.new_mes.clear()
        lmodel.training_epoch_end()
        self._logger_add_scalars(self.new_mes, self.global_epoch)
        mes.update(self.new_mes)
        return mes

    def _train(self, lmodel: LModule, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Dict[str, float]:
        model = lmodel.model
        print_model_info(model, None)
        best_mes = {}
        for _ in range(self.global_epoch + 1, self.max_epochs):
            self.global_epoch += 1
            mes = self._train_epoch(lmodel, train_dataloader)
            #
            if val_dataloader is not None:
                metrics, val_mes = self._val(lmodel, val_dataloader)
                mes.update(val_mes)
            else:
                # 只保存最后的模型, 而不保存最好的模型(用于dqn).
                #   不使用train_loss作为best_ckpt的原因: train_loss一定随着epoch增加而越来越小. 所以不需要.
                metrics = None
            is_best = self._epoch_end(mes, metrics)  # 保存模型和results
            if is_best:
                best_mes = mes
        return best_mes

    @torch.no_grad()
    def _val(self, lmodel: LModule, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        #
        metrics = 0.  # sum stat
        mes = {}
        with tqdm(total=len(dataloader), desc="  Val: ") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.new_mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                _m = lmodel.validation_step(batch)
                metrics += _m.item() if isinstance(_m, Tensor) else _m
                self._add_new_mes(mes, self.new_mes)
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, self.new_mes, self.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                if batch_idx % self.prog_bar_n_steps == 0:
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        self._logger_add_scalars(mes, self.global_epoch)
        return metrics / len(dataloader), mes

    @ torch.no_grad()
    def _test(self, lmodel: LModule, dataloader: DataLoader) -> Dict[str, float]:
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        #
        mes = {}  # sum stat
        with tqdm(total=len(dataloader), desc="Test: ") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.new_mes.clear()
                batch = lmodel.batch_to_device(batch, self.device)
                lmodel.test_step(batch)
                self._add_new_mes(mes, self.new_mes)  # LModule.log的内容
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, self.new_mes, self.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                if batch_idx % self.prog_bar_n_steps == 0:
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        self._logger_add_scalars(mes, self.global_epoch)
        return mes

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Dict[str, float]:
        # 主要内容: 记录训练的过程, 包括train每个迭代的信息; val每个epoch的信息
        device_r = next(self.lmodel.model.parameters()).device
        best_mes = self._train(self.lmodel, train_dataloader, val_dataloader)
        self.lmodel.model.to(device_r)
        return best_mes

    def test(self, dataloader: DataLoader, model_type: Literal["last", "best"] = "last") -> Dict[str, float]:
        # 主要内容: 不需要记录信息, 只需要打印测试结果即可.
        if model_type == "best":
            assert self.best_ckpt_path is not None
            self.lmodel.load_from_checkpoint(self.best_ckpt_path)
        elif model_type != "last":
            raise ValueError(f"model_type: {model_type}")

        device_r = next(self.lmodel.model.parameters()).device
        mes = self._test(self.lmodel, dataloader)
        self.lmodel.model.to(device_r)
        #
        if model_type == "best":  # 复原
            assert self.last_ckpt_path is not None
            self.lmodel.load_from_checkpoint(self.last_ckpt_path)
        return mes

# 更多的examples见 `https://github.com/Jintao-Huang/ml_alg/blob/main/examples`
# if __name__ == "__main__":
#     import torch.nn as nn
#     import torch.optim as optim
#     try:
#         from . import MLP_L2, XORDataset, accuracy_score, seed_everything
#     except ImportError:
#         from _trash import MLP_L2, XORDataset
#         from metrics import accuracy_score
#         from utils import seed_everything
#     #
#     seed_everything(4)
#     train_dataset = XORDataset(512)
#     val_dataset = XORDataset(256)
#     test_dataset = XORDataset(256)
#     ldm = LDataModule(train_dataset, val_dataset, test_dataset, 64)

#     #
#     model = MLP_L2(2, 4, 1)
#     optimizer = optim.SGD(model.parameters(), 0.1, 0.9)

#     class MyLModule(LModule):
#         def __init__(self, model: Module, optim: Optimizer, loss_fn: Module, lr_s: LRScheduler) -> None:
#             super(MyLModule, self).__init__(model, optim, {"model": "MLP_2"})
#             self.loss_fn = loss_fn
#             self.lr_s = lr_s

#         def epoch_end(self) -> None:
#             # fit. 用于lr_schedules的处理
#             self.log("lr0", self.lr_s.get_last_lr()[0])
#             self.lr_s.step()

#         def training_step(self, batch: Any) -> Tensor:
#             x_batch, y_batch = batch
#             y = self.model(x_batch)[:, 0]
#             loss: Tensor = self.loss_fn(y, y_batch.float())
#             self.log("train_loss", loss)
#             return loss

#         def validation_step(self, batch: Any) -> Union[Tensor, float]:
#             x_batch, y_batch = batch
#             y = self.model(x_batch)[:, 0]
#             y = y >= 0
#             acc = accuracy_score(y, y_batch)
#             self.log("val_acc", acc)
#             return acc

#         def test_step(self, batch: Any) -> None:
#             x_batch, y_batch = batch
#             y = self.model(x_batch)[:, 0]
#             y = y >= 0
#             acc = accuracy_score(y, y_batch)
#             self.log("test_acc", acc)
#     #
#     _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
#     RUNS_DIR = os.path.join(_ROOT_DIR, "runs")
#     os.makedirs(RUNS_DIR, exist_ok=True)
#     #
#     runs_dir = os.path.join(RUNS_DIR, "test_mini_pl")
#     loss_fn = nn.BCEWithLogitsLoss()
#     lr_s = MultiStepLR(optimizer, [10, 50], 0.1)
#     lmodel = MyLModule(model, optimizer, loss_fn, lr_s)
#     #
#     trainer = Trainer(lmodel, 'cuda', 100, runs_dir)
#     trainer.fit(ldm.train_dataloader, ldm.val_dataloader)
#     trainer.test(ldm.test_dataloader)
#     trainer.test(ldm.test_dataloader, "best")
