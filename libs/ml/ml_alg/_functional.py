# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from typing import Tuple, Union
from torch import Tensor


__all__ = []


"""exp造成的数值不稳定问题"""


def logsumexp(x: Tensor, dim: Union[Tuple[int, ...], int], keepdim: bool = False):
    """logsumexp是数值不稳定的, 容易exp时出现inf. 可以将e^x全部除以e^{max(x)}解决."""
    x_max = x.max()
    return (x-x_max).exp_().sum(dim, keepdim).log_() + x_max


def logsumexp_bad(x: Tensor, dim: Union[Tuple[int, ...], int], keepdim: bool = False):
    return x.exp().sum(dim, keepdim).log_()


# if __name__ == "__main__":
#     x = torch.tensor([1, 2, 3, 500.])
#     print(torch.logsumexp(x, -1))
#     print(logsumexp(x, -1))
#     print(logsumexp_bad(x, -1))

def softmax(x: Tensor, dim: Union[Tuple[int, ...], int]):
    """e^x/{归一化}. exp, sumexp是数值不稳定的, 会产生inf, nan. 因softmax为分式, 可以通过上下同除以e^{max(x)}解决"""
    x_max = x.max()
    x = x.sub(x_max)
    res = x.exp()
    return res.div_(res.sum(dim))


def softmax_bad(x: Tensor, dim: Union[Tuple[int, ...], int]):
    """e^x/{归一化}"""
    res = x.exp()
    return res.div_(res.sum(dim))


# if __name__ == "__main__":
#     x = torch.tensor([1, 2, 3, 500.])
#     print(torch.softmax(x, -1))
#     print(softmax(x, -1))
#     print(softmax_bad(x, -1))
