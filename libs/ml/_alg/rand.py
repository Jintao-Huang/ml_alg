import torch
from typing import Tuple, Union
import numpy as np

Size = Union[Tuple[int, ...], int]


def normal(mean: float, std: float, size: Size):
    return torch.randn(size) * std + mean


def uniform(a: float, b: float, size: Size):
    return torch.rand(size) * (b - a) + a


def randperm(n: int):
    res = torch.arange(n)
    for i in range(n):
        x = torch.randint(i, n, (), dtype=torch.long)
        res[[i, x], ] = res[[x, i], ]  # swap
    return res


if __name__ == "__main__":
    x = normal(10, 4, 1000)
    print(x.mean(), x.std())
    x = uniform(4, 10, 1000)
    print(x.min(), x.max())
    #
    torch.manual_seed(42)
    print(torch.randperm(10))
    torch.manual_seed(42)
    print(randperm(10))
