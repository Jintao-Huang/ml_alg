# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import List, Tuple, Any, Dict, Optional
import random
import torch
import numpy as np
import torch.cuda as cuda
import time
from typing import Optional, Callable, Tuple, List, Dict, Any
from torch import Tensor
from collections import defaultdict
from numpy import ndarray
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

__all__ = ["split_dataset", "stat", "test_time", "seed_everything", "time_synchronize",
           "remove_keys", "gen_seed_list", "multi_runs",
           #
           "freeze_layers", "print_model_info",
           "label_smoothing_cross_entropy", "fuse_conv_bn", "fuse_linear_bn"]


def split_dataset(dataset: Dataset, n_list: List[int], split_keys: List[str], seed: int = 42) -> List[Dataset]:
    """将数据集切分为多个数据集. (使用随机切分)
    n_list: [800, 100, 100]. 则切成3份
    split_keys: 需要切分的keys. e.g. ["data", "targets"]. 注意: 会把v: list转成ndarray
    """
    assert len(n_list) >= 2
    d_len = len(dataset)
    random_state = np.random.RandomState(seed)
    perm_idxs = random_state.permutation(d_len)
    #
    res = []
    keys = dataset.__dict__
    idx = 0
    for n in n_list:
        new_dataset = dataset.__new__(dataset.__class__)
        pos = slice(idx, idx + n)
        idx += n
        #
        for k in keys:
            v = getattr(dataset, k)
            if k in split_keys:
                if isinstance(v, list):
                    v = np.array(v)  # note!
                v = v[perm_idxs[pos]]
            setattr(new_dataset, k, v)
        res.append(new_dataset)
    return res


def stat(x: ndarray) -> Tuple[Tuple[float, float, float, float], str]:
    """统计. 返回: (mean, std, max_, min_), stat_str"""
    mean = x.mean().item()
    std = x.std().item()
    max_ = x.max().item()
    min_ = x.min().item()
    stat_str = f"{mean:.6f}±{std:.6f} |max: {max_:.6f} |min: {min_:.6f}"
    return (mean, std, max_, min_), stat_str


def test_time(func: Callable[[], Any], number: int = 1, warm_up: int = 0,
              timer: Optional[Callable[[], float]] = None) -> Any:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    res = None
    # 预热
    for _ in range(warm_up):
        res = func()
    #
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    # 打印平均, 标准差, 最大, 最小
    ts = np.array(ts)
    _, stat_str = stat(ts)
    # print
    logger.info(
        f"time[number={number}]: {stat_str}")
    return res


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_deterministic"""
    # 返回seed
    if seed is None:
        # seed_min = np.iinfo(np.uint32).min
        seed_max = np.iinfo(np.uint32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if gpu_dtm is True:
        # https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        # True: cudnn只选择deterministic的卷积算法
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)  # 会报错
        # True: cuDNN从多个卷积算法中进行benchmark, 选择最快的
        # 若deterministic=True, 则benchmark一定为False
        torch.backends.cudnn.benchmark = False
    logger.info(f"Global seed set to {seed}")
    return seed


def time_synchronize() -> float:
    # 单位: 秒
    cuda.synchronize()
    return time.perf_counter()


def remove_keys(state_dict: Dict[str, Any], prefix_keys: List[str]) -> Dict[str, Any]:
    """将带某前缀的keys删除. 不是inplace的. 应用: load_state_dict时"""
    res = {}
    for k, v in state_dict.items():
        need_saved = True
        for pk in prefix_keys:
            if k.startswith(pk):
                need_saved = False
                break
        if need_saved:
            res[k] = v
    return res


if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *

# if __name__ == "__main__":
#     # test seed_everything
#     s = seed_everything(3234335211)
#     print(s)
#     # test time_synchronize
#     x = torch.randn(10000, 10000, device='cuda')
#     res = test_time(lambda: x@x, 10, 0, time_synchronize)
#     print(res[1, :100])


def gen_seed_list(n: int, seed: Optional[int] = None,) -> List[int]:
    max_ = np.iinfo(np.uint32).max
    random_state = np.random.RandomState(seed)
    return random_state.randint(0, max_, n).tolist()


def multi_runs(collect_res: Callable[[int], Dict[str, float]], n: int, seed: Optional[int] = None, *,
               seed_list: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """跑n次的结果.
    collect_res: 函数: 传入seed, 返回result.
    n: 跑的次数. {seed_list的优先级更高, 若提供seed_list, 则n, seed无效}
    """
    t = time.perf_counter()
    if seed_list is None:
        seed_list = gen_seed_list(n, seed)
    n = len(seed_list)
    result: Dict[str, List] = defaultdict(list)
    for _seed in seed_list:
        _res = collect_res(_seed)
        for k, v in _res.items():
            result[k].append(v)
    t = int(time.perf_counter() - t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    t = f"{h:02d}:{m:02d}:{s:02d}"
    # 计算mean, std等.
    res: Dict[str, Dict[str, Any]] = {}
    res_str: List = []
    res_str.append(
        f"[RUNS_MES] n_runs: {n} |time: {t} |seed_list: {seed_list}"
    )
    res["runs_mes"] = {
        "n_runs": n,
        "time": t,
        "seed_list": seed_list
    }
    for k, v_list in result.items():
        v_list = np.array(v_list)
        (mean, std, max_, min_), stat_str = stat(v_list)
        res_str.append(
            f"  {k}: {stat_str}")
        res[k] = {
            "mean": mean,
            "std": std,
            "max_": max_,
            "min_": min_,
        }
    logger.info("\n".join(res_str))
    return res


def freeze_layers(model: Module, layer_prefix_names: List[str]) -> None:
    """inplace"""
    for n, p in model.named_parameters():
        requires_grad = True
        for lpn in layer_prefix_names:
            if n.startswith(lpn):
                requires_grad = False
                break
        p.requires_grad_(requires_grad)


def print_model_info(model: Module, inputs: Optional[Tuple[Any, ...]] = None) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # FLOPs
    #
    n_params /= 1e6
    n_grads /= 1e6
    s = [
        f"{model.__class__.__name__}: ",
        f"{len(list(model.modules()))} layers, ",
        f"{n_params:.4f}M parameters, ",
        f"{n_grads:.4f}M grads",
    ]
    if inputs is not None:
        from thop import profile
        macs, _ = profile(deepcopy(model), inputs, verbose=False)
        flops = macs * 2
        flops /= 1e9
        s += f", {flops:.4f}G FLOPs"

    logger.info("".join(s))


# if __name__ == "__main__":
#     from torchvision.models import resnet50
#     import torch

#     model = resnet50()
#     input = torch.randn(1, 3, 224, 224)
#     print_model_info(model, (input, ))
#     print_model_info(model)
#     print_model_info(model, (input, ))


def label_smoothing_cross_entropy(pred: Tensor, target: Tensor,
                                  smoothing: float = 0.01) -> Tensor:
    """
    pred: [N, F]. Tensor[float]. 未过softmax
    target: [N]. Tensor[long]
    smoothing: 若smoothing为0.1, 则target=4, n_labels=5, 对应:
        [0.02, 0.02, 0.02, 0.02, 0.92]
    """
    n_labels = pred.shape[1]
    # 构造target. 将target->[N, F]. ，target[i]的第target和样本设为1-smoothing.
    # 然后加上smoothing / n_labels
    res: Tensor = F.one_hot(target, n_labels)  # long
    res = res * (1-smoothing)
    res.add_(smoothing / n_labels)
    # 计算loss
    res.mul_(F.log_softmax(pred, dim=-1))
    return -res.sum() / pred.shape[0]


if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *


# if __name__ == "__main__":
#     x = torch.randn((1000, 100))
#     x2 = torch.randint(0, 100, (1000,))
#     y3 = libs_ml.test_time(lambda: F.cross_entropy(x, x2), number=10)
#     y = libs_ml.test_time(lambda: label_smoothing_cross_entropy(x, x2, smoothing=0.9), number=10, warm_up=5)
#     print(y, y3)


@torch.no_grad()
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """返回的Conv是freeze(require_grad=False)的. 因为不能被继续训练了."""
    # 组合conv和batchnorm, 变成一层. 从而加快infer的速度
    # 一张图片的一小块经过卷积, 并经过Bn可以规约为 只经过一层卷积
    # [N, C, KH, KW] -> [N, C2, KHO, KWO] -> [N, C2, KHO, KWO]
    # bn: 对C2做归一化. weight, bias: [C2]
    ###
    # W2 = weight / sqrt(var + eps)
    # B2 = bias - mean * scale
    # bn(x) = W2 * x + B2
    # 将W2变为对角矩阵. W2 @ (Wx+B) + B2 -> W2@W@x + W2@B+B2
    new_conv = nn.Conv2d(conv.in_channels, conv.out_channels,
                         conv.kernel_size, conv.stride,
                         conv.padding, conv.dilation, conv.groups, True)
    device = conv.weight.device
    new_conv.requires_grad_(False).to(device=device)
    W_shape = conv.weight.shape
    # [Cout, Cin, KH, KW] -> [Cout, -1]. 对一个图像块[N, Cin*KH*KW]而言, 就是过了全连接.
    W = conv.weight.view(W_shape[0], -1)
    B = conv.bias
    W2 = bn.weight * (bn.running_var + bn.eps).rsqrt_()
    B2 = (-bn.running_mean).mul_(W2).add_(bn.bias)
    W2 = torch.diag(W2)
    W_new = (W2 @ W).view(*W_shape)
    B_new = B2
    if B is not None:
        B_new.add_(W2 @ B)
    new_conv.weight.copy_(W_new)
    new_conv.bias.copy_(B_new)
    return new_conv


# if __name__ == "__main__":
#     libs_ml.seed_everything(42)
#     conv = nn.Conv2d(16, 32, 3, 1, 1, bias=True).to('cuda')
#     bn = nn.BatchNorm2d(32, 1e-5).to('cuda')
#     x = torch.randn(3, 16, 28, 28).to('cuda')
#     bn.eval()
#     y = bn(conv(x))
#     conv2 = fuse_conv_bn(conv, bn)
#     y2 = conv2(x)
#     print(torch.allclose(y, y2, atol=1e-6))

@torch.no_grad()
def fuse_linear_bn(linear: nn.Linear, bn: nn.BatchNorm1d):
    # 组合linear和batchnorm, 变成一层. 从而加快infer的速度
    # 同理fuse_conv_bn
    new_linear = nn.Linear(linear.in_features, linear.out_features, True)
    device = linear.weight.device
    new_linear.requires_grad_(False).to(device=device)
    W = linear.weight
    B = linear.bias
    W2 = bn.weight * (bn.running_var + bn.eps).rsqrt_()
    B2 = (-bn.running_mean).mul_(W2).add_(bn.bias)
    W2 = torch.diag(W2)
    W_new = (W2 @ W)
    B_new = B2
    if B is not None:
        B_new.add_(W2 @ B)
    new_linear.weight.copy_(W_new)
    new_linear.bias.copy_(B_new)
    return new_linear


# if __name__ == "__main__":
#     libs_ml.seed_everything(42)
#     linear = nn.Linear(16, 32, bias=True).to('cuda')
#     bn = nn.BatchNorm1d(32, 1e-5).to('cuda')
#     x = torch.randn(3, 16).to('cuda')
#     bn.eval()
#     y = bn(linear(x))
#     new_linear = fuse_linear_bn(linear, bn)
#     y2 = new_linear(x)
#     print(torch.allclose(y, y2, atol=1e-6))
