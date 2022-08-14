# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import List, Callable
import math
__all__ = ["cosine_annealing_lr",
           "WarmupCosineAnnealingLR", "WarmupCosineAnnealingLR2"]


def _get_offset_func(fa: float, fb: float, ga: float, gb: float) -> Callable[[float], float]:
    """将y=[sa..sb]的曲线 -> y=[ta..tb]. 曲线的趋势不变"""
    # 存在f, g; 已知: g(x)=s(f(x)+t), 求s,a. 返回func: f->g. s,a为标量
    # 即: 通过缩放和平移, 将f->g
    # s(f(a)+t)=g(a); s(f(b)+t)=g(b)
    if fa == fb:
        raise ValueError("fa == fb")
    if ga == gb:
        return lambda x: ga
    s = (ga-gb) / (fa-fb)
    t = ga / s - fa

    def func(x):
        return s * (x + t)
    return func


def cosine_annealing_lr(epoch: int, T_max: int, eta_min: float, initial_lrs: List[float]) -> List[float]:
    if epoch % (2 * T_max) == 0:
        return initial_lrs
    if (epoch + T_max) % (2 * T_max) == 0:
        # 一般最后的lr是不使用的. step()后退出循环了. 这里只是形式.
        return [eta_min] * len(initial_lrs)
    # 余弦曲线
    #   epoch=0: lr=initial_lr
    #   epoch=T_max: lr=eta_min
    # 周期为T_max * 2的cos函数: 系数=2pix/T
    res = []
    x = math.cos(math.pi * epoch / T_max)
    # 缩放[-1, 1] -> [eta_min, initial_lr]
    for initial_lr in initial_lrs:
        func = _get_offset_func(-1, 1, eta_min, initial_lr)
        res.append(func(x))
    return res


class _CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0., last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super(_CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self, last_epoch=None) -> List[float]:
        if last_epoch is None:
            last_epoch = self.last_epoch  # 应用: WarmupCosineAnnealingLR2
        return cosine_annealing_lr(last_epoch, self.T_max, self.eta_min, self.base_lrs)


# warmup1和2的区别:
#   1: 已有lr_schedular曲线, 然后将warmup之前的曲线进行缩放. 即: 不会改变warmup后的lr_schedular曲线. (也有大量实验使用1)
#   2. warmup是独立的. 升到initial_lr后再进行 lr_schedular. (huggingface使用2)
#   注意: 使用warmup后, 使用iter作为step的单位. T_max=max_epoch * len(dataloader)


class WarmupCosineAnnealingLR(_CosineAnnealingLR):
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        """例如一共有10步. warmup=3, T_max还是=10"""
        # warmup一般使用iter_idx(epoch)作为T_max进行控制
        # initial_lr -> eta_min
        self.warmup = warmup
        super(WarmupCosineAnnealingLR, self).__init__(
            optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> List[float]:
        # lr_s.step()含两部分: self.last_epoch += 1; get_lr()
        #
        # epoch/iter从1开始, 避免第一步lr为0. 也使最后一步lr=eta_min
        last_epoch = self.last_epoch + 1
        lrs = super(WarmupCosineAnnealingLR, self).get_lr(last_epoch)
        scale = 1
        if last_epoch < self.warmup:
            scale = last_epoch / self.warmup
            # scale = (self.last_epoch + 1) / self.warmup
        return [lr * scale for lr in lrs]


class WarmupCosineAnnealingLR2(_CosineAnnealingLR):
    # warmup会导致T_max的缩小.
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        """例如一共有10步. warmup=3, 则T_max应该=7"""
        # warmup一般使用iter_idx(epoch)作为T_max进行控制
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR2, self).__init__(
            optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> List[float]:
        # epoch/iter从1开始, 避免第一步lr为0. 也使最后一步lr=eta_min
        last_epoch = self.last_epoch + 1
        if last_epoch < self.warmup:
            scale = last_epoch / self.warmup  # k=1/self.warmup
            return [lr * scale for lr in self.base_lrs]
        lrs = super(WarmupCosineAnnealingLR2, self).get_lr(
            last_epoch - self.warmup)
        return lrs


# if __name__ == "__main__":
#     import torch
#     from torch.optim.sgd import SGD
#     from torch.nn.parameter import Parameter
#     from torch.optim.lr_scheduler import CosineAnnealingLR
#     param = [Parameter(torch.randn(100,))]
#     #
#     optim = SGD(param, 1e-2)
#     lrs = CosineAnnealingLR(optim, 10, 1e-4)
#     for i in range(20):
#         print(i, lrs.get_last_lr())
#         optim.step()
#         lrs.step()
#     print(20, lrs.get_last_lr())
#     #
#     optim = SGD(param, 1e-2)
#     lrs = WarmupCosineAnnealingLR(optim, 3, 10, 1e-4)
#     for i in range(20):
#         print(i, lrs.get_last_lr())
#         optim.step()
#         lrs.step()
#     #
#     optim = SGD(param, 1e-2)
#     lrs = WarmupCosineAnnealingLR2(optim, 3, 7, 1e-4)
#     for i in range(17):
#         print(i, lrs.get_last_lr())
#         optim.step()
#         lrs.step()
