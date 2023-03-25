# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ..._types import *
__all__ = ["normal", "uniform"]

Size = Union[Tuple[int, ...], int]


def _normal(mean: float, std: float, size: Size) -> Tensor:
    return torch.randn(size).mul_(std).add_(mean)


def _uniform(a: float, b: float, size: Size) -> Tensor:
    return torch.rand(size).mul_(b - a).add_(a)


def normal(
    mean: float, std: float,
    size: Size,
    dtype: Dtype = torch.float32,
    device: Device = Device("cpu"),
    generator: Optional[TGenerator] = None,
) -> Tensor:
    return torch.empty(size, dtype=dtype, device=device).normal_(mean, std, generator=generator)


def uniform(
    a: float, b: float,
    size: Size,
    dtype: Dtype = torch.float32,
    device: Device = Device("cpu"),
    generator: Optional[TGenerator] = None,
) -> Tensor:
    return torch.empty(size, dtype=dtype, device=device).uniform_(a, b, generator=generator)


# if __name__ == "__main__":
#     from libs import *
#     x = _normal(10, 4, 1000)
#     print(x.mean(), x.std())
#     #
#     ml.seed_everything(42)
#     x = _normal(10, 4, 1)
#     ml.seed_everything(42)
#     x2 = torch.normal(10, 4, (1,))
#     ml.seed_everything(42)
#     x3 = torch.distributions.Normal(10, 4).sample()
#     print(x, x2, x3)
#     #
#     x = _uniform(4, 10, 1000)
#     print(x.min(), x.max())


if __name__ == "__main__":
    import mini_lightning as ml
    ml.seed_everything(42)
    y = ml.test_time(lambda: normal(0.1, 0.2, (10000, 1000)), 10)
    ml.seed_everything(42)
    y2 = ml.test_time(lambda: torch.normal(0.1, 0.2, (10000, 1000)), 10)
    print(torch.allclose(y, y2))


def randperm(n: int) -> Tensor:
    res = torch.arange(n)
    for i in range(n):
        x = torch.randint(i, n, (), dtype=torch.long)
        res[[i, x], ] = res[[x, i], ]  # swap
    return res


# if __name__ == "__main__":
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
    L = tl.cholesky(cov)  # [F, F]
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
#     ml.seed_everything(42)
#     print(m.sample())
#     #
#     m = MultivariateNormal(mu, scale_tril=tl.cholesky(sigma))
#     ml.seed_everything(42)
#     print(m.sample())
#     #
#     ml.seed_everything(42)
#     x = multivariate_normal(mu, sigma)[0]
#     print(x)
#     x = multivariate_normal(mu, sigma, 10000)
#     print(x.mean(dim=0), x.t().cov())
