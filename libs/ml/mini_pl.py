# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
try:
    from .utils import print_model_info, select_device
except ImportError:
    from utils import print_model_info, select_device
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
import logging
from torch.cuda.amp.grad_scaler import GradScaler
from torch import autocast
from torch.nn.parallel import DataParallel
import re

logger = logging.getLogger(__name__)

# 未来会添加的功能: 多GPU, 断点续训
#   暂时取消对GAN的支持.
# 约定: epoch_idx/global_epoch, batch_idx从0开始. global_step从1开始.
__all__ = ["LModule", "LDataModule", "Trainer"]


class LModule:
    def __init__(self, model: Module, optimizer: Optimizer,
                 hparams: Optional[Dict[str, Any]] = None) -> None:
        """hparams: 需要保存的超参数. """
        # 一般: 定义loss_fn, lrs. (optim, model)
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
        """只在training_step, validation_step, test_step函数中log才有效. 
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
        # 已过loss.backward
        # note: amp导致的不优化情况(e.g. grad中含nan), 会导致lrs的UserWarning警告.
        self.trainer.scaler.step(self.optimizer)

    def training_step(self, batch: Any) -> Tensor:
        # [train]
        # 返回的Tensor(loss)用于优化
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
    def __init__(self, lmodel: LModule, device: List[int], max_epochs: int, runs_dir: str, *,
                 n_accumulate_grad: int = 1, amp: bool = False,
                 log_every_n_steps: int = 5, prog_bar_n_steps: int = 1,
                 gradient_clip_norm: Optional[float] = None, benchmark: Optional[bool] = None) -> None:
        """
        device: 若传入多个device, 则使用DP. (暂时不支持DDP). 
            e.g. []: 代表"cpu", [0], [0, 1, 2]
        n_accumulate_grad: 梯度累加. 使用mean累加, 而不是sum. 
            (使用sum可能会对weight_decay, gradient_clip_norm的取值产生影响)
            与多gpu训练效果基本一致的. 
            在不使用BN的情况下, 与batch_size*n_accumulate_grad训练的效果基本一致. 
            note: 因为会改变optimizer_step()调用的频率. 所以适当调整optimizer_step()中调用的lr_scheduler的参数. (e.g. warmup, T_max等)
            note: 增大了total_batch_size可以适当增加学习率
        amp: 是否使用混合精度训练. 
            作用: 加快训练速度, 减少显存消耗. 略微(或不)下降性能 (因为可以提高batch size). 
            note: 建议在大型/超大型模型中使用. 小模型并不会加快训练速度. (有些环境可能不支持amp)
        log_every_n_steps: 几步需要将信息log入tensorboard. 使用global_step % log_every_n_steps. 
            这不会修改prog_bar(进度条)的显示(每个step都会更新). 
        prog_bar_n_steps: 进度条的显示的频率
        gradient_clip_norm: 梯度裁剪, 防止梯度爆炸(INF, NAN)
        benchmark: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            Pytorch默认False. 该函数的benchmark行为与Pytorch Lightning行为一致 
            True: 可以加速训练, 但是会造成不可复现. 
        """
        self.lmodel = lmodel
        self.lmodel.trainer = self
        self.device = select_device(device)
        if len(device) > 1:
            lmodel.model = DataParallel(lmodel.model)
        self.max_epochs = max_epochs
        self.n_accumulate_grad = n_accumulate_grad
        self.amp = amp
        self.log_every_n_steps = log_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.gradient_clip_norm = gradient_clip_norm
        #
        time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S.%f")
        v = self._get_version(runs_dir)
        runs_dir = os.path.join(runs_dir, f"v{v}-{time}")
        logger.info(f"runs_dir: {runs_dir}")
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
        self.scaler = GradScaler(enabled=amp)
        self.best_metrics = -1e10  # model save
        self.best_epoch_idx: int = None
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

    @staticmethod
    def _get_version(runs_dir):
        fnames = os.listdir(runs_dir)
        v_list = [-1]
        for fname in fnames:
            m = re.match(r"v(\d+)", fname)
            if m is None:
                continue
            v = m.group(1)
            v_list.append(int(v))
        return max(v_list) + 1

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
            res = repr(hparams)
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
            self.best_epoch_idx = self.global_epoch
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
        scaler = self.scaler
        #
        mes = {}
        #
        with tqdm(total=len(dataloader) // self.n_accumulate_grad,
                  desc=f"Epoch {self.global_epoch}") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.global_step += 1
                batch = lmodel.batch_to_device(batch, self.device)
                self.new_mes.clear()
                with autocast(device_type=self.device.type, enabled=self.amp):
                    loss = lmodel.training_step(batch)
                    loss.div_(self.n_accumulate_grad)
                # log
                # log lr
                for i, lr in enumerate([group['lr'] for group in lmodel.optimizer.param_groups]):
                    lmodel.log(f"lr{i}", lr, prog_bar_mean=False)
                self._add_new_mes(mes, self.new_mes)
                # tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(self.new_mes, self.global_step)
                #
                scaler.scale(loss).backward()
                # 优化
                if self.global_step % self.n_accumulate_grad == 0:
                    if self.gradient_clip_norm is not None:
                        # grad裁剪需要下面这行.
                        scaler.unscale_(lmodel.optimizer)
                        # amp=True的nan情况会自行调节scaler, error_if_nonfinite=False
                        clip_grad_norm_(
                            model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=not self.amp)
                    lmodel.optimizer_step()
                    scaler.update()
                    # set_to_none可以增加时间/内存性能. 该行为与Pytorch Lightning默认行为不一致
                    lmodel.optimizer.zero_grad(set_to_none=True)
                    #
                    mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                    log_mes = self._get_log_mes(
                        mean_mes, self.new_mes, self.prog_bar_mean)
                    prog_bar.set_postfix(log_mes, refresh=False)
                    if (batch_idx + 1) // self.n_accumulate_grad % self.prog_bar_n_steps == 0:
                        prog_bar.update(self.prog_bar_n_steps)

            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        # 后处理
        lmodel.training_epoch_end()
        return mes

    def _train(self, lmodel: LModule, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Dict[str, float]:
        model = lmodel.model
        print_model_info(model, None)
        best_mes: Dict[str, float] = {}
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
                best_mes.update({"global_epoch": self.global_epoch,
                                 "global_step": self.global_step})
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
                batch = lmodel.batch_to_device(batch, self.device)
                self.new_mes.clear()
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
    def _test(self, lmodel: LModule, dataloader: DataLoader, *, model_type: Literal["last", "best"] = "last") -> Dict[str, float]:
        model = lmodel.model
        #
        model.eval()
        model.to(self.device)
        #
        mes = {}  # sum stat
        desc = "Test Last: " if model_type == "last" else "Test Best: "
        with tqdm(total=len(dataloader), desc=desc) as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                batch = lmodel.batch_to_device(batch, self.device)
                self.new_mes.clear()
                lmodel.test_step(batch)
                self._add_new_mes(mes, self.new_mes)  # LModule.log的内容
                #
                mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                log_mes = self._get_log_mes(
                    mean_mes, self.new_mes, self.prog_bar_mean)
                prog_bar.set_postfix(log_mes, refresh=False)
                if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        epoch_idx = self.global_epoch if model_type == "last" else self.best_epoch_idx
        # 要先前再后, 不然会覆盖.
        self._logger_add_scalars(mes, epoch_idx)
        return mes

    def fit(self, train_dataloader: DataLoader, val_dataloader: DataLoader) -> Dict[str, float]:
        """返回val中metrics最好的log信息(含train和val). """
        device_r = next(self.lmodel.model.parameters()).device
        best_mes = self._train(self.lmodel, train_dataloader, val_dataloader)
        self.lmodel.model.to(device_r)
        return best_mes

    @staticmethod
    def key_add_suffix(mes: Dict[str, Any], suffix: str) -> Dict[str, Any]:
        """not inplace
        suffix: 建议以`_`开头
        """
        res = {}
        for k, v in mes.items():
            res[k + suffix] = v
        return res

    def test(self, dataloader: DataLoader) -> Dict[str, float]:
        """返回best, last model的test的log信息"""
        # test "best"
        mes = {}
        device_r = next(self.lmodel.model.parameters()).device
        if self.best_ckpt_path is not None:  # 复原
            assert self.last_ckpt_path is not None  # 一般都满足
            self.lmodel.load_from_checkpoint(self.best_ckpt_path)
            mes = self._test(self.lmodel, dataloader, model_type="best")
            mes = self.key_add_suffix(mes, "_best")
            self.lmodel.load_from_checkpoint(self.last_ckpt_path)
        # test "last"
        mes2 = self._test(self.lmodel, dataloader, model_type="last")
        mes2 = self.key_add_suffix(mes2, "_last")
        mes.update(mes2)
        self.lmodel.model.to(device_r)
        return mes


# 更多的examples见 `https://github.com/Jintao-Huang/ml_alg/blob/main/examples`
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    try:
        from . import MLP_L2, XORDataset, accuracy_score, seed_everything
    except ImportError:
        from _trash import MLP_L2, XORDataset
        from metrics import accuracy_score
        from utils import seed_everything
    #
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s: %(filename)s:%(lineno)d] %(message)s ")  
    seed_everything(2)
    train_dataset = XORDataset(512)
    val_dataset = XORDataset(256)
    test_dataset = XORDataset(256)
    ldm = LDataModule(train_dataset, val_dataset, test_dataset, 64)

    #
    model = MLP_L2(2, 4, 1)
    optimizer = optim.SGD(model.parameters(), 0.1, 0.9)

    class MyLModule(LModule):
        def __init__(self, model: Module, optim: Optimizer, loss_fn: Module, lr_s: LRScheduler) -> None:
            super(MyLModule, self).__init__(model, optim, {"model": "MLP_2"})
            self.loss_fn = loss_fn
            self.lr_s = lr_s

        def training_epoch_end(self) -> None:
            # fit. 用于lr_schedules的处理
            self.lr_s.step()

        def training_step(self, batch: Any) -> Tensor:
            x_batch, y_batch = batch
            y = self.model(x_batch)[:, 0]
            loss: Tensor = self.loss_fn(y, y_batch.float())
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
    loss_fn = nn.BCEWithLogitsLoss()
    lr_s = MultiStepLR(optimizer, [10, 50], 0.1)
    lmodel = MyLModule(model, optimizer, loss_fn, lr_s)
    #
    trainer = Trainer(lmodel, [], 100, runs_dir)
    logger.info(trainer.fit(ldm.train_dataloader, ldm.val_dataloader))
    logger.info(trainer.test(ldm.test_dataloader))
