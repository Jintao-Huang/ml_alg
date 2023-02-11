# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import math
import os
import random
import time
import logging
from copy import deepcopy
from typing import List, Tuple, Any, Dict, Optional, Literal
from typing import Optional, Callable, Tuple, List, Dict, Any, Union
from collections import defaultdict
#
from tqdm import tqdm
import numpy as np
from numpy import ndarray
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda as cuda
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor, device as Device
from torch.utils.data import Dataset
from torch.nn.parallel import DataParallel as DP, DistributedDataParallel as DDP
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
#
from torchmetrics import Metric
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.models import resnet152
import mini_lightning as ml


__all__ = ["load_state_dict", "save_state_dict",
           "split_dataset", "extract_dataset", "smart_load_state_dict",
           "fuse_conv_bn", "fuse_linear_bn", "test_metric", "reserve_memory", "test_nan"]

logger = ml.logger
#


def load_state_dict(model: Module, fpath: str) -> IncompatibleKeys:
    device = next(model.parameters()).device
    return model.load_state_dict(torch.load(fpath, device))


def save_state_dict(model: Module, fpath: str) -> None:
    torch.save(model.state_dict(), fpath)


def extract_dataset(dataset: Dataset, idxs: Union[slice, List[int], ndarray], split_keys: List[str]) -> Dataset:
    """
    idxs: ndarray可以是ndarray[bool]等.
    """
    keys = dataset.__dict__
    new_dataset = dataset.__new__(dataset.__class__)
    #
    for k in keys:
        v = getattr(dataset, k)
        if k in split_keys:
            if isinstance(v, list):
                v = np.array(v)  # note!
            v = v[idxs]
        setattr(new_dataset, k, deepcopy(v))
    return new_dataset


def split_dataset(dataset: Dataset, n_list: List[int], split_keys: List[str],
                  shuffle: bool = True, seed: int = 42) -> List[Dataset]:
    """将数据集切分为多个数据集. (使用随机切分)
    n_list: [800, 100, 100]. 则切成3份
    split_keys: 需要切分的keys. e.g. ["data", "targets"]. 注意: 会把v: list转成ndarray
    shuffle: 是否随机切分
    seed: 只有shuffle的情况下, 才用到
    """
    d_len = len(dataset)
    if shuffle:
        random_state = np.random.RandomState(seed)
        perm_idxs = random_state.permutation(d_len)
    #
    res = []
    idx = 0
    for i, n in enumerate(n_list):
        if i == len(n_list) - 1 and n == -1:
            n = d_len - idx
        pos = slice(idx, idx + n)
        if shuffle:
            pos = perm_idxs[pos]
        idx += n
        new_dataset = extract_dataset(dataset, pos, split_keys)
        res.append(new_dataset)
    return res


if __name__ == "__main__":
    from libs import *


def smart_load_state_dict(model: Module, state_dict: Dict[str, Tensor],
                          prefix_key: str = "", strict: bool = True) -> IncompatibleKeys:
    if prefix_key != "":
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[prefix_key + k] = v
        state_dict = new_state_dict
    #
    return model.load_state_dict(state_dict, strict=strict)


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


def test_metric(metric: Metric, *args: Tensor, return_mes: bool = False) -> Tuple[Tensor, List[Tensor]]:
    # args: preds, target
    td = TensorDataset(*args)
    N = args[0].shape[0]
    loader = DataLoader(td, batch_size=math.ceil(N/5), shuffle=True)
    mes = []
    for batch_args in loader:
        # metric.update(*batch_args)
        mes.append(metric(*batch_args))
    if return_mes:
        return metric.compute(), mes
    else:
        return metric.compute()


# if __name__ == "__main__":
#     import mini_lightning as ml
#     from torchmetrics import MeanMetric
#     from torchmetrics.classification.accuracy import Accuracy
#     from torchmetrics.functional.classification.accuracy import accuracy
#     ml.seed_everything(1, False)
#     preds = torch.randint(0, 10, (17,), dtype=torch.long)
#     target = torch.randint(0, 10, (17,), dtype=torch.long)
#     acc_metric = Accuracy("multiclass", num_classes=10)
#     acc = test_metric(acc_metric, preds, target, return_mes=True)
#     acc2 = accuracy(preds, target, "multiclass", num_classes=10)
#     print(acc, acc2)
#     #
#     loss = torch.randint(0, 10, (17,), dtype=torch.float32)
#     mean_metric = MeanMetric()
#     mean = test_metric(mean_metric, loss, return_mes=True)
#     mean2 = loss.mean()
#     print(mean, mean2)


def reserve_memory(device_ids: List[int], max_G: int = 9) -> None:
    """占用显存, 防止被抢"""
    ml.select_device(device_ids)
    for d in range(len(device_ids)):
        device = Device(d)
        model = resnet152().to(device)
        for i in tqdm(range(128)):
            image = torch.randn(i, 3, 224, 224).to(device)
            try:
                model(image)
            except RuntimeError:
                break
            gb = cuda.memory_reserved(device) / 1e9
            if gb >= max_G:
                break
            cuda.empty_cache()
        #
        logger.info(f"reserved memory: {cuda.memory_reserved(device) / 1e9:.4f}GB")


# if __name__ == "__main__":
#     reserve_memory([0], 5)


def _test_object_nan(arr: ndarray) -> ndarray:
    assert arr.dtype == np.object_
    res = np.zeros_like(arr, dtype=np.bool8)
    for i, x in enumerate(arr):
        if isinstance(x, float) and math.isnan(x):
            res[i] = True
    return np.array(res)


def test_nan(arr: ndarray) -> ndarray:
    if arr.dtype == np.object_:
        res = _test_object_nan(arr)
    else:
        res = np.isnan(arr)
    n = np.sum(res)
    logger.info(f"N(NAN): {n}")
    return res


if __name__ == "__main__":
    x = np.array([1, 1, float("nan"), float("nan")])
    print(test_nan(x))
    x = np.array([1, "1", float("nan"), float("nan")], dtype=np.object_)
    print(test_nan(x))
