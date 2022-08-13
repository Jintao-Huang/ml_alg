# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
try:
    from .utils import print_model_info, select_device
except ImportError:
    # for debug
    from utils import print_model_info, select_device
import torch
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
import re
import torch.distributed as dist

logger = logging.getLogger(__name__)

# 未来会添加的功能: resume_from_checkpoint, ddp, auto_lr_find, sync_bn
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
        note: lr会自动进行log, 无需手动log. 
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
        return

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
        # note: 第一步的amp/found_inf导致的不优化情况, 可能会导致lrs的UserWarning警告.
        if self.trainer.amp or not self.trainer.found_inf:
            # 在amp=False情况下, 使用`self.optimizer.step()`是一样的.
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

    def __init__(
        self, train_dataset: Optional[Dataset], val_dataset: Optional[Dataset], test_dataset: Optional[Dataset],
        batch_size: int, 
        num_workers: int = 0,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
        *,
        drop_last_train: bool = True,
        shuffle_train: bool = True,
        pin_memory_train: bool = True
    ) -> None:

        self.train_dataloader: DataLoader = None
        self.val_dataloader: DataLoader = None
        self.test_dataloader: DataLoader = None
        #
        if train_dataset:
            self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=shuffle_train,
                                               num_workers=num_workers, pin_memory=pin_memory_train,
                                               drop_last=drop_last_train, collate_fn=collate_fn)
        for dataset, loader_name in zip([val_dataset, test_dataset], ["val_dataloader", "test_dataloader"]):
            if dataset:
                loader = DataLoader(dataset, batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=False,
                                    drop_last=False, collate_fn=collate_fn)
                setattr(self, loader_name, loader)


class Trainer:
    def __init__(
        self, lmodel: LModule, device_ids: List[int],
        max_epochs: int, runs_dir: str,
        n_accumulate_grad: int = 1,
        amp: bool = False,
        gradient_clip_norm: Optional[float] = None,
        *,
        log_every_n_steps: int = 5,
        prog_bar_n_steps: int = 1,
        benchmark: Optional[bool] = None
    ) -> None:
        """
        device_ids: 若传入多个device_ids, 则使用DP. (暂时不支持DDP). 
            e.g. []: 代表"cpu", [0], [0, 1, 2]
            note: DP: 推荐在大型/超大型模型中使用. 小模型并不会加快训练速度. (瓶颈在gpu之间的通信)
                batch_size会被拆分到各个gpu中. 请确保batch_size % n_gpus == 0. 
            note: DP会对lmodel.model进行赋值: `lmodel.model = DP(lmodel.model)`
        n_accumulate_grad: 梯度累加. 
            使用mean累加, 而不是sum. 
                Refer: https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loops/optimization/optimizer_loop.html (搜索: self.trainer.accumulate_grad_batches)
                (使用sum可能会对weight_decay, gradient_clip_norm的取值产生影响)
            与多gpu训练效果基本一致的. 
                在不使用BN的情况下, 与batch_size*n_accumulate_grad训练的效果基本一致. 
            note: 因为会改变optimizer_step()调用的频率. 所以适当调整optimizer_step()中调用的lr_scheduler的参数. (e.g. warmup, T_max等)
            note: 增大了total_batch_size可以适当增加学习率
            note: 使用batch_idx % , 批次最后的未更新的grad会到epoch结束时更新. 与pytorch lightning行为相同. 
        amp: 是否使用混合精度训练. 
            作用: 加快训练速度, 减少显存消耗. 略微(或不)下降性能 (因为可以提高batch size). 
            note: 推荐在大型/超大型模型中使用. 小模型并不会加快训练速度. (有些环境可能不支持amp)
            Refer: https://pytorch.org/docs/stable/notes/amp_examples.html
        gradient_clip_norm: 梯度裁剪(norm裁剪), 防止梯度爆炸. 一般设置为5, 10, 20.
            note: 在梯度裁剪的基础上加了INF的检查. 这可以提高训练的稳定性. 
                若在梯度裁剪中发现INF. 则会跳过本次更新. (amp=True情况下, 此检查由amp处理)
        *
        log_every_n_steps: 几步需要将信息log入tensorboard. 使用global_step % . 
        prog_bar_n_steps: 进度条的显示的频率. batch_idx % .
        benchmark: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            Pytorch默认False. 若该函数的benchmark行为与Pytorch Lightning行为一致 
            benchmark=True: 可以加速训练, 但是会造成不可复现. 
            benchmark=None: 若cudnn.deterministic为False, 则设置为True. 否则, 设为False. 
                note: deterministic也可以通过libs_ml.seed_everything中的参数gpu_dtm指定. (这里不设置参数)
        """
        self.lmodel = lmodel
        self.lmodel.trainer = self
        self.device_ids = device_ids
        self.device = select_device(device_ids)
        if len(device_ids) > 1:
            lmodel.model = DP(lmodel.model)

        logger.info(f"Using DP: {len(device_ids) > 1}")
        self.max_epochs = max_epochs
        self.n_accumulate_grad = n_accumulate_grad
        self.amp = amp
        if amp:
            logger.info(f"Using amp: {amp}")

        self.log_every_n_steps = log_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.benchmark = benchmark
        #
        time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")  # .%f
        v = self._get_version(runs_dir)
        runs_dir = os.path.join(runs_dir, f"v{v}-{time}")
        logger.info(f"runs_dir: {runs_dir}")
        #
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
        deterministic = torch.backends.cudnn.deterministic
        if deterministic:
            benchmark = False
        else:
            benchmark = True if benchmark is None else benchmark
        torch.backends.cudnn.benchmark = benchmark
        logger.info(
            f"Setting benchmark: {benchmark}")
        #
        self.logger = SummaryWriter(self.tb_dir)
        self.scaler = GradScaler(enabled=amp)
        self.best_metrics = -1e10  # model save
        self.best_epoch_idx: int = -1
        self.best_ckpt_path: str = ""
        self.last_ckpt_path: str = ""
        self.global_step = 0
        self.global_epoch = -1
        #
        hparams = self.lmodel.hparams
        self.save_hparams(hparams)
        # 用于log. 含义见LModule.log
        self.new_mes: Dict[str, float] = {}
        self.prog_bar_mean: Dict[str, bool] = {}
        # 用于梯度裁剪中, 对于found_inf的处理. 跳过本次更新. (amp=True情况下不工作. amp会对inf进行处理.)
        self.found_inf = False

    @staticmethod
    def _get_version(runs_dir):
        if os.path.isdir(runs_dir):
            fnames = os.listdir(runs_dir)
        else:
            fnames = []
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
            logger.info(f"Saving hparams: {saved_hparams}")
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
        if mode == "best" and self.best_ckpt_path:
            os.remove(self.best_ckpt_path)
        elif mode == "last" and self.last_ckpt_path:
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
            print((f"- best model, saving model `{ckpt_fname}`"))
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
        with tqdm(total=len(dataloader),
                  desc=f"Epoch {self.global_epoch}") as prog_bar:
            for batch_idx, batch in enumerate(dataloader):
                self.global_step += 1
                batch = lmodel.batch_to_device(batch, self.device)
                self.new_mes.clear()
                with autocast(device_type=self.device.type, enabled=self.amp):
                    loss = lmodel.training_step(batch)
                # log lr
                for i, lr in enumerate([group['lr'] for group in lmodel.optimizer.param_groups]):
                    lmodel.log(f"lr{i}", lr, prog_bar_mean=False)
                self._add_new_mes(mes, self.new_mes)
                # tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(self.new_mes, self.global_step)
                #
                loss.div_(self.n_accumulate_grad)
                scaler.scale(loss).backward()
                # 优化
                if (batch_idx + 1) % self.n_accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
                    if self.gradient_clip_norm is not None:
                        # grad裁剪需要下面这行.
                        scaler.unscale_(lmodel.optimizer)
                        found_inf = clip_grad_norm_(
                            model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=False)
                        if not self.amp:  # amp=True情况, found_inf下不工作. amp会对inf进行处理.
                            self.found_inf = found_inf.isinf().all().item()
                    lmodel.optimizer_step()
                    scaler.update()
                    # set_to_none可以增加速度. 该行为与Pytorch Lightning默认行为不一致.
                    # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                    lmodel.optimizer.zero_grad(set_to_none=True)
                    self.found_inf = False
                # prog_bar
                if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                    mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                    log_mes = self._get_log_mes(
                        mean_mes, self.new_mes, self.prog_bar_mean)
                    prog_bar.set_postfix(log_mes, refresh=False)
                    prog_bar.update(self.prog_bar_n_steps)

            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        # 后处理
        lmodel.training_epoch_end()
        return mes

    def _train(self, lmodel: LModule, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        if len(self.device_ids) > 1:
            # DP并不会因为 无法平均拆分inputs而崩溃. 但这里为了规范性, 进行检查.
            assert train_dataloader.batch_size % len(self.device_ids) == 0
            assert val_dataloader is None or val_dataloader.batch_size % len(
                self.device_ids) == 0
        model = lmodel.model
        print_model_info(model, None)
        best_mes: Dict[str, float] = {}
        mes = {}
        #
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
            mes.update({"global_epoch": self.global_epoch,
                        "global_step": self.global_step})
            if is_best:
                best_mes = mes
        if not best_mes:
            best_mes = mes  # last
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
                # prog_bar
                if batch_idx % self.prog_bar_n_steps == 0:
                    mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                    log_mes = self._get_log_mes(
                        mean_mes, self.new_mes, self.prog_bar_mean)
                    prog_bar.set_postfix(log_mes, refresh=False)
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        self._logger_add_scalars(mes, self.global_epoch)
        return metrics / len(dataloader), mes

    @ torch.no_grad()
    def _test(self, lmodel: LModule, dataloader: DataLoader, *, model_type: Literal["last", "best"] = "last") -> Dict[str, float]:
        model = lmodel.model
        if len(self.device_ids) > 1:
            assert dataloader.batch_size % len(self.device_ids) == 0
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
                # prog_bar
                if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                    mean_mes = self._sum_to_mean(mes, batch_idx + 1)
                    log_mes = self._get_log_mes(
                        mean_mes, self.new_mes, self.prog_bar_mean)
                    prog_bar.set_postfix(log_mes, refresh=False)
                    prog_bar.update(self.prog_bar_n_steps)
            prog_bar.update(prog_bar.total - prog_bar.n)
        self._sum_to_mean(mes, len(dataloader), inplace=True)
        epoch_idx = self.global_epoch if model_type == "last" else self.best_epoch_idx
        self._logger_add_scalars(mes, epoch_idx)
        return mes

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
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

    def test(self, dataloader: DataLoader, only_best: bool = True) -> Dict[str, float]:
        """返回best, last model的test的log信息
        only_best: 只测试best. 理论上测试集不能作为验证集的作用使用. 所以默认为True. 
        """
        # note: 若先last, 后best, 则last会在tensorboard中被覆盖. 所以这里先best, 后last.
        # test "best"
        mes = {}
        device_r = next(self.lmodel.model.parameters()).device
        if self.best_ckpt_path:  # 复原
            assert self.last_ckpt_path  # 一般都满足
            self.lmodel.load_from_checkpoint(self.best_ckpt_path)
            mes = self._test(self.lmodel, dataloader, model_type="best")
            mes = self.key_add_suffix(mes, "_best")
            self.lmodel.load_from_checkpoint(self.last_ckpt_path)
        # test "last"
        if not only_best:
            mes2 = self._test(self.lmodel, dataloader, model_type="last")
            mes2 = self.key_add_suffix(mes2, "_last")
            mes.update(mes2)
        self.lmodel.model.to(device_r)
        return mes


# 更多的examples见 `https://github.com/Jintao-Huang/ml_alg/blob/main/examples`
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    #
    from _trash import MLP_L2, XORDataset
    from metrics import accuracy_score
    from utils import seed_everything
    #
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s: %(filename)s:%(lineno)d] %(message)s ")
    seed_everything(2, gpu_dtm=True)
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
    runs_dir = os.path.join(RUNS_DIR, "test_mini_lightning")
    loss_fn = nn.BCEWithLogitsLoss()
    lr_s = MultiStepLR(optimizer, [10, 50], 0.1)
    lmodel = MyLModule(model, optimizer, loss_fn, lr_s)
    #
    trainer = Trainer(lmodel, [], 100, runs_dir)
    logger.info(trainer.fit(ldm.train_dataloader, None))
    logger.info(trainer.test(ldm.test_dataloader, False))
