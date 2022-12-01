# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import torch
from typing import Tuple, Union
from torch import Tensor
__all__ = []

Size = Union[Tuple[int, ...], int]


def normal(mean: float, std: float, size: Size) -> Tensor:
    return torch.randn(size) * std + mean


def uniform(a: float, b: float, size: Size) -> Tensor:
    return torch.rand(size) * (b - a) + a


def randperm(n: int) -> Tensor:
    res = torch.arange(n)
    for i in range(n):
        x = torch.randint(i, n, (), dtype=torch.long)
        res[[i, x], ] = res[[x, i], ]  # swap
    return res


# if __name__ == "__main__":
#     x = normal(10, 4, 1000)
#     print(x.mean(), x.std())
#     x = uniform(4, 10, 1000)
#     print(x.min(), x.max())
#     #
#     torch.manual_seed(42)
#     print(torch.randperm(10))
#     torch.manual_seed(42)
#     print(randperm(10))


# if __name__ == "__main__":
#     x = torch.randn(100)
#     keep_tensors = torch.bernoulli(torch.full_like(x, 0.5))
#     keep_tensors2 = torch.randint_like(x, 0, 2)
#     print(keep_tensors, keep_tensors2)


def multivariate_normal(mean: Tensor, cov: Tensor, N: int = 1) -> Tensor:
    """
    mean: Tensor[F] {F个随机变量}
    cov: Tensor[F, F]
    n: 共采样n次
    return: [N, F]
    """
    F = mean.shape[0]
    std_norm = torch.randn(N, F)
    L = torch.linalg.cholesky(cov)  # [F, F]
    return (std_norm @ L).add_(mean)


# if __name__ == "__main__":
#     from torch.distributions import MultivariateNormal
#     from libs import *
#     mu = torch.tensor([1, 2.])
#     sigma = torch.tensor([
#         [1, 1.],
#         [1, 2],
#     ])
#     m = MultivariateNormal(mu, sigma)
#     libs_ml.seed_everything(42)
#     print(m.sample())
#     #
#     m = MultivariateNormal(mu, scale_tril=torch.linalg.cholesky(sigma))
#     libs_ml.seed_everything(42)
#     print(m.sample())
#     #
#     libs_ml.seed_everything(42)
#     x = multivariate_normal(mu, sigma)[0]
#     print(x)
#     x = multivariate_normal(mu, sigma, 10000)
#     print(x.mean(dim=0), x.t().cov())
