# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import torch
from typing import Tuple, Union
from torch import Tensor
from torch import dtype as Dtype

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


def var(X: Tensor, unbiased: bool = True) -> Tensor:
    """
    X: [...]
    return: []
    """
    X = X.ravel()  # [N]
    mean = X.mean()  # []
    diff = X - mean
    ddof = X.shape[0] - unbiased
    return torch.einsum("i,i->", diff, diff).div_(ddof)


if __name__ == "__main__":
    import mini_lightning as ml

# if __name__ == "__main__":
#     x = torch.randn(10000, 1000)
#     y = ml.test_time(lambda: torch.var(x), 10)
#     y2 = ml.test_time(lambda: var(x), 10)
#     print(torch.allclose(y, y2))


def cov(X: Tensor, correction: int = 1) -> Tensor:
    """
    X: [N, F]
    return [N, N]
    """
    F = X.shape[1]
    mean = X.mean(dim=1, keepdim=True)  # [N]
    diff = X - mean
    res = (diff @ diff.T).div_(F - correction)
    return res

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: torch.cov(x), 10)
#     y2 = ml.test_time(lambda: cov(x), 10)
#     print(torch.allclose(y, y2))


def corrcoef(X: Tensor) -> Tensor:
    """
    X: [N, F]
    return: [N, N]
    """
    res = torch.cov(X)  # correction 随意
    std = res.diag().sqrt_()
    res.div_(std[:, None]).div_(std)
    return res


# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: torch.corrcoef(x), 10)
#     y2 = ml.test_time(lambda: corrcoef(x), 10)
#     print(torch.allclose(y, y2))


def bincount(X: Tensor, n_bins: int = -1, dtype: Dtype = torch.long) -> Tensor:
    """
    X: long. [N]
    return: [n_bins]. long
    """
    device = X.device
    if n_bins == -1:
        n_bins = int(X.max().item()) + 1
    res = torch.zeros((n_bins), dtype=dtype, device=device)
    res.index_put_((X,), torch.tensor(1, dtype=dtype, device=device), accumulate=True)
    return res


# if __name__ == "__main__":
#     X = torch.randint(0, 1000, (1000000,))
#     y = ml.test_time(lambda: bincount(X, 1000), 10)
#     y2 = ml.test_time(lambda: torch.bincount(X, minlength=1000), 10)
#     print(torch.allclose(y, y2))
