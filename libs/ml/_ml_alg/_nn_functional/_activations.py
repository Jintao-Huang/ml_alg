# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:


from ...._types import *
# from libs import *


def threshold(x: Tensor, threshold: Union[int, float], value: Union[int, float]) -> Tensor:
    x = x.clone()
    x[x < threshold] = value
    return x

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: threshold(x, 1, 2), 10)
#     y2 = ml.test_time(lambda: F.threshold(x, 1, 2), 10)
#     print(torch.allclose(y, y2))


def tanh(x: Tensor) -> Tensor:
    """(e^x-e^{-x})/(e^x+e^{-x})
    or (1-e^{-2x})/(1+e^{-2x})"""
    _2x_neg_exp = x.mul(2).neg_().exp_()
    return (1 - _2x_neg_exp).div_(1 + _2x_neg_exp)


# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: tanh(x), 10)
#     y2 = ml.test_time(lambda: F.tanh(x), 10)
#     print(torch.allclose(y, y2, atol=1e-6))


def sigmoid(x: Tensor) -> Tensor:
    """1/(1+e^{-x})"""
    x_neg_exp = x.neg().exp_()
    return x_neg_exp.add_(1).reciprocal_()


# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: sigmoid(x), 10)
#     y2 = ml.test_time(lambda: F.sigmoid(x), 10)
#     print(torch.allclose(y, y2, atol=1e-6))

def softmax(x: Tensor, dim: int) -> Tensor:
    x_exp = x.exp()
    return x_exp.div(x_exp.sum(dim=dim, keepdim=True))


# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: softmax(x, 1), 10)
#     y2 = ml.test_time(lambda: F.softmax(x, 1), 10)
#     print(torch.allclose(y, y2, atol=1e-6))


def silu(x: Tensor, inplace: bool = False) -> Tensor:
    """x*sigmoid(x)"""
    if inplace:
        x0 = x.clone()
        return x.sigmoid_().mul_(x0)
    else:
        return x.sigmoid().mul_(x)

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: silu(x), 10)
#     y2 = ml.test_time(lambda: F.silu(x), 10)
#     x2 = x.clone()
#     y3 = ml.test_time(lambda: silu(x2, inplace=True), 10)
#     x2 = x.clone()
#     y4 = ml.test_time(lambda: F.silu(x2, inplace=True), 10)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y3, y4))


def _std_gaussian_cdf(x: Tensor) -> Tensor:
    """not inplace"""
    return (torch.erf_(x.div(math.sqrt(2))).add_(1)).div_(2)


def gelu(x: Tensor, approximate: Literal["none", "tanh"] = "none") -> Tensor:
    """x*phi(x). phi(x)是高斯分布的CDF{近似于sigmoid(1.702x)}"""
    if approximate == "none":
        return _std_gaussian_cdf(x).mul_(x)
    elif approximate == "tanh":
        tmp = math.sqrt(2/math.pi) * (x.add(x.pow(3), alpha=0.044715))
        return x.div(2).mul_(tmp.tanh_().add_(1))
    else:
        raise ValueError(f"approximate: {approximate}")

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: gelu(x), 10)
#     y2 = ml.test_time(lambda: F.gelu(x), 10)
#     x2 = x.clone()
#     y3 = ml.test_time(lambda: gelu(x2, approximate="tanh"), 10)
#     x2 = x.clone()
#     y4 = ml.test_time(lambda: F.gelu(x2, approximate="tanh"), 10)
#     print(torch.allclose(y, y2, atol=1e-6))
#     print(torch.allclose(y3, y4, atol=1e-6))


def leaky_relu(x: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if not inplace:
        x = x.clone()
    x[x < 0] *= negative_slope  # .mul_()会有问题.
    return x

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     y = ml.test_time(lambda: leaky_relu(x), 10)
#     y2 = ml.test_time(lambda: F.leaky_relu(x), 10)
#     x2 = x.clone()
#     y3 = ml.test_time(lambda: leaky_relu(x2, inplace=True), 10)
#     x2 = x.clone()
#     y4 = ml.test_time(lambda: F.leaky_relu(x2, inplace=True), 10)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y3, y4))
