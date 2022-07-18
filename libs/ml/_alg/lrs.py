from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler, CosineAnnealingLR
from torch.optim import SGD, Optimizer
from torch.nn.parameter import Parameter
import torch
from typing import List, Callable
import math


class _MultiStepLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, milestones: List[int],
                 gamma: float = 0.1, last_epoch: int = -1) -> None:
        self.milestones = set(milestones)
        self.gamma = gamma
        super(_MultiStepLR, self).__init__(optimizer, last_epoch)
        # self.last_epoch: int
        # self.optimizer.param_groups: List[Dict[str, Any]].
        #   groups[0]['lr']: float
        #   groups[0]['initial_lr']{别名: self.base_lrs}

    # self.get_last_lr()  # 返回List[float]
    def get_lr(self) -> List[float]:
        epoch = self.last_epoch
        lr_group = self.optimizer.param_groups
        if epoch in self.milestones:
            scale = self.gamma
        else:
            scale = 1
        return [group['lr'] * scale for group in lr_group]


def get_offset_func(fa: float, fb: float, ga: float, gb: float) -> Callable[[float], float]:
    """将y=[sa..sb]的曲线 -> y=[ta..tb]"""
    # 存在fx, gx; 已知: gx=s(fx+a), 求s,a. 返回func: fx->gx
    # s(fa+a)=ga; s(fb+a)=gb
    if fa == fb:
        raise ValueError("fa == fb")
    if ga == gb:
        return lambda x: ga
    s = (ga-gb) / (fa-fb)
    a = ga / s - fa

    def func(x):
        return s * (x + a)
    return func


def cosine_annealing_lr(epoch: int, T_max: int, eta_min: float, initial_lrs: List[float]) -> List[float]:
    if epoch == 0:
        return initial_lrs
    if epoch == T_max:
        return [eta_min] * len(initial_lrs)
    if epoch > T_max:
        raise ValueError(f"epoch: {epoch}")
    # 余弦曲线
    #   epoch=0: lr=initial_lr
    #   epoch=T_max: lr=eta_min
    # 周期为T_max * 2的cos函数: 系数=2pix/T
    res = []
    x = math.cos(math.pi * epoch / T_max)
    # 缩放[-1, 1] -> [eta_min, initial_lr]
    for initial_lr in initial_lrs:
        func = get_offset_func(-1, 1, eta_min, initial_lr)
        res.append(func(x))
    return res


class _CosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0., last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super(_CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return cosine_annealing_lr(self.last_epoch, self.T_max, self.eta_min, self.base_lrs)


class _WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        # warmup一般使用iter_idx(epoch)作为T_max进行控制
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super(_WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        lrs = cosine_annealing_lr(
            self.last_epoch, self.T_max, self.eta_min, self.base_lrs)
        scale = 1
        if self.last_epoch <= self.warmup:
            scale = self.last_epoch / self.warmup
        return [lr * scale for lr in lrs]


class _WarmupCosineAnnealingLR2(_LRScheduler):
    # 这个warmup: 0: 0; warmup: initial_lr; T_max: eta_min
    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        # warmup一般使用iter_idx(epoch)作为T_max进行控制
        self.warmup = warmup
        self.T_max = T_max
        self.eta_min = eta_min
        super(_WarmupCosineAnnealingLR2, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:

        if self.last_epoch <= self.warmup:
            scale = self.last_epoch / self.warmup
            return [lr * scale for lr in self.base_lrs]
        else:
            return cosine_annealing_lr(self.last_epoch - self.warmup, self.T_max - self.warmup, self.eta_min, self.base_lrs)


def test_mslr(param):
    optim = SGD(param, lr=0.01)
    lrs = MultiStepLR(optim, milestones=[5, 10], gamma=0.1)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())
    optim = SGD(param, lr=0.01)
    lrs = _MultiStepLR(optim, milestones=[5, 10], gamma=0.1)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())
    print(lrs.base_lrs)


def test_calr(param):
    optim = SGD(param, lr=0.01)
    lrs = CosineAnnealingLR(optim, 20, 1e-4)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())
    optim = SGD(param, lr=0.01)
    lrs = _CosineAnnealingLR(optim, 20, 1e-4)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())
    #
    optim = SGD(param, lr=0.01)
    lrs = _WarmupCosineAnnealingLR(optim, 10, 20, 1e-4)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())
    #
    optim = SGD(param, lr=0.01)
    lrs = _WarmupCosineAnnealingLR2(optim, 10, 20, 1e-4)
    for i in range(20):
        print(i, lrs.get_last_lr())
        optim.step()
        lrs.step()
    print(20, lrs.get_last_lr())


if __name__ == "__main__":
    param = [Parameter(torch.randn(100,))]
    # test_mslr(param)
    test_calr(param)

# 0 [0.01]
# 1 [0.009939057285945933]
# 2 [0.009757729755661013]
# 3 [0.009460482294732423]
# 4 [0.009054634122155993]
# 5 [0.008550178566873413]
# 6 [0.007959536998847746]
# 7 [0.00729725297371076]
# 8 [0.006579634122155994]
# 9 [0.005824350601949147]
# 10 [0.005050000000000002]
# 11 [0.004275649398050861]
# 12 [0.003520365877844012]
# 13 [0.0028027470262892446]
# 14 [0.0021404630011522593]
# 15 [0.0015498214331265906]
# 16 [0.001045365877844011]
# 17 [0.0006395177052675796]
# 18 [0.0003422702443389901]
# 19 [0.0001609427140540686]
# 20 [0.0001]

# 0 [0.01]
# 1 [0.009939057285945933]
# 2 [0.00975772975566101]
# 3 [0.009460482294732422]
# 4 [0.00905463412215599]
# 5 [0.008550178566873411]
# 6 [0.007959536998847742]
# 7 [0.007297252973710757]
# 8 [0.00657963412215599]
# 9 [0.005824350601949143]
# 10 [0.00505]
# 11 [0.004275649398050858]
# 12 [0.0035203658778440107]
# 13 [0.0028027470262892433]
# 14 [0.0021404630011522584]
# 15 [0.0015498214331265898]
# 16 [0.0010453658778440103]
# 17 [0.0006395177052675791]
# 18 [0.00034227024433898957]
# 19 [0.00016094271405406814]
# 20 [0.0001]

# 0 [0.0]
# 1 [0.0009939057285945933]
# 2 [0.001951545951132202]
# 3 [0.0028381446884197265]
# 4 [0.0036218536488623965]
# 5 [0.004275089283436706]
# 6 [0.004775722199308645]
# 7 [0.005108077081597529]
# 8 [0.005263707297724792]
# 9 [0.005241915541754229]
# 10 [0.00505]
# 11 [0.004275649398050858]
# 12 [0.0035203658778440107]
# 13 [0.0028027470262892433]
# 14 [0.0021404630011522584]
# 15 [0.0015498214331265898]
# 16 [0.0010453658778440103]
# 17 [0.0006395177052675791]
# 18 [0.00034227024433898957]
# 19 [0.00016094271405406814]
# 20 [0.0001]
# 0 [0.0]
# 1 [0.001]
# 2 [0.002]
# 3 [0.003]
# 4 [0.004]
# 5 [0.005]
# 6 [0.006]
# 7 [0.006999999999999999]
# 8 [0.008]
# 9 [0.009000000000000001]
# 10 [0.01]
# 11 [0.00975772975566101]
# 12 [0.00905463412215599]
# 13 [0.007959536998847742]
# 14 [0.00657963412215599]
# 15 [0.00505]
# 16 [0.0035203658778440107]
# 17 [0.0021404630011522584]
# 18 [0.0010453658778440103]
# 19 [0.00034227024433898957]
# 20 [0.0001]
