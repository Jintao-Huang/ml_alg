# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
try:
    from .utils import print_model_info, select_device, remove_keys, de_parallel, en_parallel, smart_load_state_dict
    from ..utils import save_to_yaml
except ImportError:
    # for debug main
    from utils import print_model_info, select_device, remove_keys, de_parallel, en_parallel, smart_load_state_dict
    import sys
    import os
    #
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs.utils import save_to_yaml

import torch
from torch import device as Device, Tensor
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from torch.nn import Module, SyncBatchNorm
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler as LRScheduler
import torch.cuda as cuda

import os
from typing import List, Any, Dict, Optional, Tuple, Callable, Union, Sequence, Mapping, Literal
from tqdm import tqdm
import datetime
from torch.nn.utils.clip_grad import clip_grad_norm_
import logging
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
import re
import torch.distributed as dist
from bisect import bisect_right


# 使用torchrun 启动DDP. https://pytorch.org/docs/stable/elastic/run.html.
# 例子见: examples/cv_ddp.py
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

logger = logging.getLogger(__name__)

# 未来会添加的功能: 自动调参
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
        """
        prog_bar_mean: 在prog_bar中显示的是整个epoch的均值. (一般loss, acc用均值. lr不用均值)
        note: lr会自动进行log, 无需手动log.
        """
        # 如何log. 我们调用lmodule的log, 将信息存储在lmodule中
        # 当单次迭代结束时, 会修改lmodule._mes, 随后加到trainer的全局log中...
        if RANK not in {-1, 0}:
            return
        if self.trainer is None:
            raise ValueError(f"self.trainer: {self.trainer}")
        #
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
        if RANK not in {-1, 0}:
            return
        device = next(self.model.parameters()).device
        state_dict = torch.load(ckpt_path, map_location=device)
        if self.trainer is not None and isinstance(self.model, (DP, DDP)):
            smart_load_state_dict(self.model, state_dict, "module.")
        else:
            self.model.load_state_dict(state_dict)

    def save_checkpoint(self, ckpt_path: str) -> None:
        if RANK not in {-1, 0}:
            return
        model = de_parallel(self.model)
        torch.save(model.state_dict(), ckpt_path)

    def training_epoch_start(self, device) -> None:
        # [fit]用于模型to(device), 特别是有多个model的情况下会使用(dqn)
        self.model.train()
        self.model.to(device)

    def training_epoch_end(self) -> None:
        # [fit]用于lr_schedules的处理
        #   lr的log, Trainer会自动做的.
        return

    def validation_epoch_end(self) -> Optional[float]:
        # [val]用于torchmetrics的compute.
        # log的内容会被tensorboard记录, 并返回.
        # 返回的float, 作为metrics, 作为模型的选择, 越高越好(e.g. acc, 若越低越好则可以返回负数).
        #   优先级比validation_step返回的metrics高(覆盖), None则不覆盖
        return

    def test_epoch_end(self) -> None:
        # [test]用于torchmetrics的compute.
        # log的内容会被tensorboard记录, 并返回.
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
        # note: 第一步的amp/found_inf/found_nan导致的不优化情况, 可能会导致lrs的UserWarning警告.
        if not self.trainer.found_nan and (self.trainer.amp or not self.trainer.found_inf):
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
            if dataset and RANK in {-1, 0}:
                loader = DataLoader(dataset, batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=False,
                                    drop_last=False, collate_fn=collate_fn)
                setattr(self, loader_name, loader)


class Trainer:
    def __init__(
        self, lmodel: LModule, device_ids: List[int],
        max_epochs: int, runs_dir: str,
        n_accumulate_grad: Union[int, Dict[int, int]] = 1,
        amp: bool = False,
        gradient_clip_norm: Optional[float] = None,
        sync_bn: bool = False,
        replace_sampler_ddp: bool = True,
        *,
        log_every_n_steps: int = 10,
        prog_bar_n_steps: int = 1,
        benchmark: Optional[bool] = None,
        verbose: bool = True
    ) -> None:
        """
        关于ddp mode: 不需要设置. 使用torchrun进行运行, Trainer会对此进行分辨(通过RANK/LOCAL_RANK). 例子见examples/cv_ddp.py
            note: DP mode下, train,val,test都会使用DP.
                DDP mode下, 只有train使用DDP. val,test使用single-gpu. 
                    (train时的记录的scalar只是rank=0中的指标. val/test使用单gpu, 计算的是整个数据集的metrics)
            note: 推荐使用DDP而不是DP. DDP使用多进程, DP使用多线程. DDP具有更快的训练速度. 
            warning: DDP, 作者只在single node下进行了测试. 若在multi node下实验出现问题, 请提issue. 
        # 
        device_ids: 若传入多个device_ids, 则使用DP. (暂时不支持DDP). 使用`CUDA_VISIBLE_DEVICES`环境变量进行选择
            e.g. []: 代表"cpu", [0], [0, 1, 2]
            note: DP: batch_size会被拆分到各个gpu中. 请确保batch_size % n_gpus == 0.
                DDP: 总的batch_size = world_size * batch_size. (与DP不同)
            note: DP, DDP, sync_bn会对lmodel.model进行修改.   
        n_accumulate_grad: 梯度累加.
            若为Dict[int, int]: e.g. {5:2, 20:4} or {0:1, 5:2, 20:4}. 表示前5个(0-4)epoch使用1(no累计), 5-19使用2, 20以后使用4. 
                这可以在初期加快训练速度, 并在最后不影响收敛性能. 
            使用mean累加, 而不是sum.
                Refer: https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/loops/optimization/optimizer_loop.html (搜索: self.trainer.accumulate_grad_batches)
                (使用sum可能会对weight_decay, gradient_clip_norm的取值产生影响)
            与多gpu训练效果基本一致的.
                在不使用BN的情况下, 与batch_size*n_accumulate_grad训练的效果基本一致.
            note: 因为会改变optimizer_step()调用的频率. 
                所以适当调整optimizer_step()中调用的lr_scheduler的参数. (e.g. warmup, T_max等)
                增大了total_batch_size可以适当增加学习率
            note: 使用batch_idx % , 批次最后的未更新的grad会到epoch结束时更新. 与pytorch lightning行为相同.
        amp: 是否使用混合精度训练.
            作用: 加快训练速度, 减少显存消耗. 略微(或不)下降性能 (因为可以提高batch size).
            note: 推荐在大型/超大型模型中使用. 小模型并不会加快训练速度. (有些环境可能不支持amp)
            Refer: https://pytorch.org/docs/stable/notes/amp_examples.html
        gradient_clip_norm: 梯度裁剪(norm裁剪), 防止梯度爆炸, 并对缩放前的grad_norm进行log(见verbose). 一般设置为5, 10, 20.
            note: 在梯度裁剪的基础上加了INF的检查. 这可以提高训练的稳定性.
                若在梯度裁剪中发现INF. 则会跳过本次更新. (amp=True情况下, 此检查由amp处理)
        sync_bn: (只在DDP情况下支持)同步BN. BN时, 对所有的RANK进行同步, 并统一做BN, 而不是对单个GPU做BN. 
            这通常会提高训练精度和稳定性, 但会略微降低训练速度. 
        replace_sampler_ddp: (只在DDP情况下有效)在DDP情况下, 是否使用DistributedSampler(仅train_dataloader). 
            若不使用: train时, 每个gpu都会使用完整的数据集(相当于一个epoch训练了world_size次dataset). 
            使用sampler, 可以将数据集切成world_size块, 分发给各个gpu. 
        *
        log_every_n_steps: 几步需要将信息log入tensorboard(使用每n步采样的方式, 而不是均值). 使用global_step % .
            见prog_bar_n_steps的note(同理). 
        prog_bar_n_steps: 进度条的显示的频率. batch_idx % .
            note: 在DDP+train的情况下, 使用多进程多GPU, 只采样rank=0的scalar(metrics可能并不完全准确.); 
                DDP的val, test使用单GPU. 无需担心.
        benchmark: https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
            Pytorch默认False. 若该函数的benchmark行为与Pytorch Lightning行为一致
            benchmark=True: 可以加速训练, 但是会造成不可复现.
            benchmark=None: 若cudnn.deterministic为False, 则设置为True. 否则, 设为False.
                note: deterministic也可以通过libs_ml.seed_everything中的参数gpu_dtm指定. (这里不设置参数)
        verbose: 
            True: 记录lr, 在gradient_clip_norm=True的情况下记录grad_norm. 
            False: 不记录上述内容. 使prog_bar更简洁. tensorboard仍会记录. 
        """
        logger.info(f"LOCAL_RANK: {LOCAL_RANK}, RANK: {RANK}, WORLD_SIZE: {WORLD_SIZE}")
        #
        self.lmodel = lmodel
        self.lmodel.trainer = self
        self.device_ids = device_ids
        self.device = select_device(device_ids)
        if RANK == -1:
            dp_ddp_mode = "DP" if len(device_ids) > 1 else None
        else:
            dp_ddp_mode = "DDP"
            self.device = Device(LOCAL_RANK)
            cuda.set_device(LOCAL_RANK)  # 设置当前cuda.
            assert dist.is_available()
            if not dist.is_initialized():
                backend = "nccl" if dist.is_nccl_available() else "gloo"
                logger.info(f"Using Backend: {backend}")
                dist.init_process_group(backend=backend, rank=RANK, world_size=WORLD_SIZE)
            self.lmodel.model.to(self.device)
        self.dp_ddp_mode = dp_ddp_mode
        self.sync_bn = sync_bn
        self.lmodel.model = en_parallel(self.lmodel.model, dp_ddp_mode, sync_bn)
        self.amp = amp
        logger.info(f"Using amp: {amp}")
        #
        self.max_epochs = max_epochs
        self.n_accumulate_grad = n_accumulate_grad
        if isinstance(self.n_accumulate_grad, dict):
            if 0 not in self.n_accumulate_grad.keys():
                self.n_accumulate_grad = self.n_accumulate_grad.copy()
                self.n_accumulate_grad.update({0: 1})
        self.gradient_clip_norm = gradient_clip_norm
        self.replace_sampler_ddp = replace_sampler_ddp
        #
        self.log_every_n_steps = log_every_n_steps
        self.prog_bar_n_steps = prog_bar_n_steps
        self.verbose = verbose
        #
        self.benchmark = benchmark
        deterministic = torch.backends.cudnn.deterministic
        if deterministic:
            benchmark = False
        else:
            benchmark = True if benchmark is None else benchmark
        torch.backends.cudnn.benchmark = benchmark
        logger.info(f"Setting benchmark: {benchmark}")
        #
        self.scaler = GradScaler(enabled=amp)
        self.best_metrics = -1e10  # model save
        self.best_epoch_idx: int = -1
        self.best_ckpt_path: str = ""
        self.last_ckpt_path: str = ""
        self.global_step = 0
        self.global_epoch = -1
        # 用于log. 含义见LModule.log
        self.new_mes: Dict[str, float] = {}
        self.prog_bar_mean: Dict[str, bool] = {}
        # 用于梯度裁剪中, found_inf检查. 跳过本次更新. (amp=True情况下不工作. amp会对inf进行处理.)
        self.found_inf = False
        # amp不检查nan. 这里对在梯度裁剪中, found_nan检查. 跳过本次更新 (0/0 inf/inf会产生nan)
        self.found_nan = False
        #
        if RANK in {-1, 0}:
            time = datetime.datetime.now().strftime("%Y:%m:%d-%H:%M:%S")  # .%f
            v = self._get_version(runs_dir)
            runs_dir = os.path.join(runs_dir, f"v{v}-{time}")
            logger.info(f"runs_dir: {runs_dir}")
            #
            self.ckpt_dir = os.path.join(runs_dir, "checkpoints")
            self.tb_dir = os.path.join(runs_dir, "runs")  # tensorboard
            self.hparams_path = os.path.join(runs_dir, "hparams.yaml")
            self.result_path = os.path.join(runs_dir, "result.yaml")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.tb_dir, exist_ok=True)
            #
            self.tb_logger = SummaryWriter(self.tb_dir)
            hparams = self.lmodel.hparams
            self.save_hparams(hparams)

    @staticmethod
    def _get_version(runs_dir: str) -> int:
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

    def _check_hparams(self, hparams: Any) -> Any:
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
                res.append(self._check_hparams(hp))
        elif isinstance(hparams, Mapping):
            res = {}
            for k, v in hparams.items():
                res[k] = self._check_hparams(v)
        else:
            res = repr(hparams)
        return res

    def save_hparams(self, hparams: Dict[str, Any]) -> None:
        saved_hparams = self._check_hparams(hparams)
        logger.info(f"Saving hparams: {saved_hparams}")
        save_to_yaml(saved_hparams, self.hparams_path)

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
            self.tb_logger.add_scalar(k, v, global_step=step)

    def _remove_ckpt(self, mode: str) -> None:
        if mode == "best" and self.best_ckpt_path:
            os.remove(self.best_ckpt_path)
        elif mode == "last" and self.last_ckpt_path:
            os.remove(self.last_ckpt_path)

    def _epoch_end(self, mes: Dict[str, float], metric: Optional[float]) -> bool:
        # 1. 模型保存
        is_best = False
        if metric is not None and metric >= self.best_metrics:  # 含等于
            # 保存
            self._remove_ckpt("best")
            self.best_metrics = metric
            ckpt_fname = f"best-epoch={self.global_epoch}-metrics={metric}.ckpt"
            self.best_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
            self.best_epoch_idx = self.global_epoch
            self.lmodel.save_checkpoint(self.best_ckpt_path)
            print((f"- best model, saving model `{ckpt_fname}`"))
            is_best = True
        #
        self._remove_ckpt("last")
        ckpt_fname = f"last-epoch={self.global_epoch}-metrics={metric}.ckpt"
        self.last_ckpt_path = os.path.join(self.ckpt_dir, ckpt_fname)
        self.lmodel.save_checkpoint(self.last_ckpt_path)
        # 2. 结果保存
        save_to_yaml({f"Epoch={self.global_epoch}": mes}, self.result_path, mode="a")
        return is_best

    def _add_new_mes(self, mes: Dict[str, float], new_mes: Dict[str, float], alpha: int) -> None:
        # alpha: 系数: 一个batch中的N
        for k, v in new_mes.items():
            if k not in mes:
                mes[k] = 0
            mes[k] += v * alpha

    @staticmethod
    def _get_log_mes(mean_mes: Dict[str, float], new_mes: Dict[str, float], prog_bar_mean: Dict[str, bool], verbose: bool):
        res = {}
        # 假设mes, new_mes, prog_bar_mean的k相同
        if not verbose:
            mean_mes = remove_keys(mean_mes, ["lr", "grad_norm"])  # no inplace
        keys = mean_mes.keys()
        for k in keys:
            if prog_bar_mean[k] is True:
                res[k] = mean_mes[k]
            else:
                res[k] = new_mes[k]
        return res

    @staticmethod
    def _get_batch_size(batch: Union[Sequence[Tensor], Mapping[str, Tensor]]) -> int:
        if isinstance(batch, Sequence):
            x = batch[0]
        elif isinstance(batch, Mapping):
            x = next(iter(batch.values()))
        return x.shape[0]

    def _train_epoch(self, lmodel: LModule, dataloader: DataLoader) -> Dict[str, float]:
        model = lmodel.model
        device = self.device
        lmodel.training_epoch_start(device)
        scaler = self.scaler
        #
        if self.replace_sampler_ddp and RANK != -1:
            dataloader.sampler.set_epoch(self.global_epoch)
        #
        if isinstance(self.n_accumulate_grad, dict):
            nag_list: List[int] = sorted(self.n_accumulate_grad.keys())  # nag: n_accumulate_grad
            idx = nag_list[bisect_right(nag_list, self.global_epoch) - 1]
            n_accumulate_grad: int = self.n_accumulate_grad[idx]
        elif isinstance(self.n_accumulate_grad, int):
            n_accumulate_grad = self.n_accumulate_grad
        else:
            raise TypeError(f"self.n_accumulate_grad: {self.n_accumulate_grad}, type: {type(self.n_accumulate_grad)}")
        #
        mes: Dict[str, float] = {}
        if RANK in {-1, 0}:
            prog_bar = tqdm(total=len(dataloader),
                            desc=f"Epoch {self.global_epoch}", mininterval=0.01,  dynamic_ncols=True)
            mes2: Dict[str, float] = {}  # optimize mes
            new_mes: Dict[str, float] = {}
            new_mes2: Dict[str, float] = {}  # optimize new mes
            N2, N3 = 0, 0  # for optim_N, for total_N
        for batch_idx, batch in enumerate(dataloader):
            self.global_step += 1
            if RANK in {-1, 0}:
                # 因为每个batch_size的大小可能不同.
                N = self._get_batch_size(batch)
                N2 += N
                N3 += N
                self.new_mes.clear()
            #
            batch = lmodel.batch_to_device(batch, device)
            with autocast(device_type=self.device.type, enabled=self.amp):
                loss = lmodel.training_step(batch)
            loss.div_(n_accumulate_grad)
            scaler.scale(loss).backward()
            #
            if RANK in {-1, 0}:
                new_mes = self.new_mes.copy()
                self._add_new_mes(mes, new_mes, N)

            # 优化
            if (batch_idx + 1) % n_accumulate_grad == 0 or (batch_idx + 1) == len(dataloader):
                if RANK in {-1, 0}:
                    self.new_mes.clear()
                if self.gradient_clip_norm:
                    # grad裁剪需要下面这行.
                    scaler.unscale_(lmodel.optimizer)
                    grad_norm = clip_grad_norm_(
                        model.parameters(), max_norm=self.gradient_clip_norm, error_if_nonfinite=False)

                    if not self.amp:  # amp=True情况, found_inf下不工作. amp会对inf进行处理.
                        self.found_inf = grad_norm.isinf().all().item()
                    self.found_nan = grad_norm.isnan().all().item()

                    # log grad_norm
                    lmodel.log(f"grad_norm", grad_norm, prog_bar_mean=False)
                # log lr
                for i, lr in enumerate([group['lr'] for group in lmodel.optimizer.param_groups]):
                    lmodel.log(f"lr{i}", lr, prog_bar_mean=False)
                #
                lmodel.optimizer_step()
                scaler.update()
                # set_to_none可以增加速度. 该行为与Pytorch Lightning默认行为不一致.
                # https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
                lmodel.optimizer.zero_grad(set_to_none=True)
                #
                self.found_inf = False
                self.found_nan = False
                if RANK in {-1, 0}:
                    new_mes2 = self.new_mes.copy()
                    self._add_new_mes(mes2, new_mes2, N2)
                    N2 = 0
            if RANK in {-1, 0}:
                # tensorboard
                if self.global_step % self.log_every_n_steps == 0:
                    self._logger_add_scalars(new_mes, self.global_step)
                    self._logger_add_scalars(new_mes2, self.global_step)
                # prog_bar
                if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                    mean_mes = self._sum_to_mean(mes, N3)
                    mean_mes2 = self._sum_to_mean(mes2, N3)
                    log_mes = self._get_log_mes(mean_mes, new_mes, self.prog_bar_mean, self.verbose)
                    log_mes2 = self._get_log_mes(mean_mes2, new_mes2, self.prog_bar_mean, self.verbose)
                    log_mes.update(log_mes2)
                    prog_bar.set_postfix(log_mes, refresh=False)
                    prog_bar.update(self.prog_bar_n_steps)
        if RANK in {-1, 0}:
            prog_bar.update(prog_bar.total - prog_bar.n)
            prog_bar.close()
            self._sum_to_mean(mes, N3, inplace=True)
            self._sum_to_mean(mes2, N3, inplace=True)
            mes.update(mes2)
            #
            mes = remove_keys(mes, ["lr", "grad_norm"])  # 不返回出去.
        # 后处理
        lmodel.training_epoch_end()
        return mes

    def _train(self, lmodel: LModule, train_dataloader: DataLoader,
               val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        if RANK == -1 and len(self.device_ids) > 1:
            # DP并不会因为 无法平均拆分inputs而崩溃. 但这里为了规范性, 进行检查.
            assert train_dataloader.batch_size % len(self.device_ids) == 0
            assert val_dataloader.batch_size % len(self.device_ids) == 0
        if self.replace_sampler_ddp and RANK != -1:
            shuffle = True
            if isinstance(train_dataloader.sampler, SequentialSampler):
                shuffle = False
            sampler = DistributedSampler(train_dataloader.dataset, shuffle=shuffle)
            logger.info(f"Using DistributedSampler")
            train_dataloader = DataLoader(train_dataloader.dataset, train_dataloader.batch_size, sampler=sampler,
                                          num_workers=train_dataloader.num_workers, pin_memory=train_dataloader.pin_memory,
                                          drop_last=train_dataloader.drop_last, collate_fn=train_dataloader.collate_fn)
        #
        model = lmodel.model
        mes: Dict[str, float] = {}
        best_mes: Dict[str, float] = {}
        print_model_info(model, None)
        #
        for _ in range(self.global_epoch + 1, self.max_epochs):
            self.global_epoch += 1
            mes = self._train_epoch(lmodel, train_dataloader)
            #
            if RANK in {-1, 0}:
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
            if RANK != -1:
                dist.barrier()
        if not best_mes:
            best_mes = mes  # last
        #
        return best_mes if RANK in {-1, 0} else {}

    def _val_test(self, lmodel: LModule, dataloader: DataLoader,
                  val_test_step: Callable[[Any], Any], val_test_epoch_end: Callable[[], Optional[float]],
                  desc: str, epoch_idx: int) -> Tuple[float, Dict[str, float]]:
        # 用于val, test
        model_r = lmodel.model
        lmodel.model = de_parallel(model_r)
        model = lmodel.model
        device = self.device
        #
        model.eval()
        model.to(device)
        N3 = 0  # for total_N
        #
        metrics = 0.  # sum stat(以dataset为单位.)
        mes: Dict[str, float] = {}
        new_mes: Dict[str, float] = {}
        prog_bar = tqdm(total=len(dataloader), desc=desc, mininterval=0.01, dynamic_ncols=True)
        for batch_idx, batch in enumerate(dataloader):
            N = self._get_batch_size(batch)
            N3 += N
            self.new_mes.clear()
            with torch.no_grad():
                batch = lmodel.batch_to_device(batch, device)
                _m = val_test_step(batch)
            if _m is not None:
                # val
                _m = _m.item() if isinstance(_m, Tensor) else _m
                metrics += _m * N
            new_mes = self.new_mes.copy()
            self._add_new_mes(mes, new_mes, N)
            # prog_bar
            if (batch_idx + 1) % self.prog_bar_n_steps == 0:
                mean_mes = self._sum_to_mean(mes, N3)
                log_mes = self._get_log_mes(mean_mes, new_mes, self.prog_bar_mean, self.verbose)
                prog_bar.set_postfix(log_mes, refresh=False)
                prog_bar.update(self.prog_bar_n_steps)
        prog_bar.update(prog_bar.total - prog_bar.n)
        prog_bar.close()
        self._sum_to_mean(mes, N3, inplace=True)
        metrics /= N3
        #
        self.new_mes.clear()
        end_metrics = val_test_epoch_end()
        if end_metrics is not None:
            metrics = end_metrics
        mes.update(self.new_mes)
        self._logger_add_scalars(mes, epoch_idx)
        #
        if isinstance(model_r, DDP):
            lmodel.model = model_r
        return metrics, mes

    def _val(self, lmodel: LModule, dataloader: DataLoader) -> Tuple[float, Dict[str, float]]:
        desc = "  Val: "
        metrics, mes = self._val_test(lmodel, dataloader, self.lmodel.validation_step, self.lmodel.validation_epoch_end,
                                      desc, self.global_epoch)
        return metrics, mes

    def _test(self, lmodel: LModule, dataloader: DataLoader,
              model_type: Literal["last", "best"]) -> Dict[str, float]:
        if RANK == -1 and len(self.device_ids) > 1:
            assert dataloader.batch_size % len(self.device_ids) == 0
        desc = "Test Last: " if model_type == "last" else "Test Best: "
        epoch_idx = self.global_epoch if model_type == "last" else self.best_epoch_idx
        _, mes = self._val_test(lmodel, dataloader, self.lmodel.test_step,  self.lmodel.test_epoch_end,
                                desc, epoch_idx)
        mes.update({"global_epoch": epoch_idx})
        return mes

    @ staticmethod
    def _key_add_suffix(mes: Dict[str, Any], suffix: str) -> Dict[str, Any]:
        """not inplace"""
        res = {}
        for k, v in mes.items():
            res[k + suffix] = v
        return res

    def fit(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader]) -> Dict[str, float]:
        """返回val中metrics最好的log信息(含train和val). """
        # best_mes除了rank in {-1, 0}. 其他都是返回 {}.
        device_r = next(self.lmodel.model.parameters()).device
        best_mes = self._train(self.lmodel, train_dataloader, val_dataloader)
        self.lmodel.model.to(device_r)
        cuda.empty_cache()
        return best_mes

    def test(self, dataloader: DataLoader, only_best: bool = True) -> Dict[str, float]:
        """返回best, last model的test的log信息
        only_best: 只测试best. 理论上测试集不能作为验证集的作用使用. 所以默认为True.
        """
        # note: 若先last, 后best, 则last会在tensorboard中被覆盖. 所以这里先best, 后last.
        # test "best"
        # mes除了rank in {-1, 0}. 其他都是返回 {}.
        device_r = next(self.lmodel.model.parameters()).device
        mes = {}
        if RANK in {-1, 0}:
            if self.best_ckpt_path:
                assert self.last_ckpt_path  # 一般都满足
                self.lmodel.load_from_checkpoint(self.best_ckpt_path)
                mes = self._test(self.lmodel, dataloader, "best")
                self.lmodel.load_from_checkpoint(self.last_ckpt_path)  # 复原
                mes = self._key_add_suffix(mes, "_best")
            # test "last"
            if not only_best:
                mes2 = self._test(self.lmodel, dataloader, "last")
                mes2 = self._key_add_suffix(mes2, "_last")
                mes.update(mes2)
        if RANK != -1:
            dist.barrier()
        self.lmodel.model.to(device_r)
        cuda.empty_cache()
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
    if not os.path.isdir(_ROOT_DIR):
        raise FileNotFoundError(f"_ROOT_DIR: {_ROOT_DIR}")
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
