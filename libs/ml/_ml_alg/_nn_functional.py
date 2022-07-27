import torch.nn.functional as F
from torch import Tensor
# 用于学习, 速度慢于F.

"""
relu: x=>0保持, x<0变为=0
leaky_relu: x>=0保持, x<0进行线性压缩(*negative_slope). 
sigmoid: 1/(1+e^x)
tanh: (e^x-e^{-x})/(e^x+e^{-x})
softmax: e^x/{归一化}
silu: x*sigmoid(x)
gelu: x*phi(x). phi(x)是高斯分布的CDF{类似于sigmoid(1.702x)}
"""


def relu(x: Tensor, inplace=False) -> Tensor:
    """
    x: Tensor[float]. [...]
    """
    if not inplace:
        return x.clamp_min(0)
    else:
        return x.clamp_min_(0)


if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *

# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     x1 = x.clone()
#     y = libs_utils.test_time(lambda: relu(x1))
#     x2 = x.clone()
#     y2 = libs_utils.test_time(lambda: relu(x2, inplace=True))
#     x3 = x.clone()
#     y3 = libs_utils.test_time(lambda: F.relu(x3))
#     x4 = x.clone()
#     y4 = libs_utils.test_time(lambda: F.relu(x4, inplace=True))
#     print(torch.allclose(y, y3))
#     print(torch.allclose(y2, y3))
#     print(torch.allclose(y3, y4))


# def _one_hot(x: Tensor, n_classes: int = -1) -> Tensor:
#     """
#     x: Tensor[long]. [N]
#     """
#     if n_classes == -1:
#         n_classes = x.max() + 1
#     return torch.eye(n_classes, dtype=torch.long, device=x.device)[x]

def one_hot(x: Tensor, n_classes: int = -1) -> Tensor:
    """推荐.
    x: Tensor[long]. [N]
    """
    if n_classes == -1:
        n_classes = x.max() + 1
    res = torch.zeros((x.shape[0], n_classes),
                      dtype=torch.long, device=x.device)
    res[torch.arange(x.shape[0]), x] = 1
    return res


# if __name__ == "__main__":
#     x = torch.randint(0, 10, (1000,))  # long
#     # y = libs_utils.test_time(lambda: _one_hot(x), number=10)
#     y2 = libs_utils.test_time(lambda: one_hot(x), number=10)
#     y3 = libs_utils.test_time(lambda: F.one_hot(x), number=10)
#     # print(torch.allclose(y, y2))
#     print(torch.allclose(y2, y3))

def nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: Tensor[float]. [N, F]
    target: Tensor[long]. [N]
    """
    n_labels = pred.shape[1]
    target = F.one_hot(target, n_labels)  # long
    res = pred * target
    return -res.sum() / pred.shape[0]


# if __name__ == "__main__":
#     x = torch.randn((1000, 10))
#     x2 = torch.randint(0, 10, (1000,))
#     y = libs_utils.test_time(lambda: nll_loss(x, x2))
#     y2 = libs_utils.test_time(lambda: F.nll_loss(x, x2))
#     print(y, y2)
#     print(torch.allclose(y, y2))


def cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: Tensor[float]. [N, F]
    target: Tensor[long]. [N]
    """

    return F.nll_loss(F.log_softmax(pred, dim=1), target)

# if __name__ == "__main__":
#     x = torch.randn((1000, 10))
#     x2 = torch.randint(0, 10, (1000,))
#     y = libs_utils.test_time(lambda: cross_entropy(x, x2))
#     y2 = libs_utils.test_time(lambda: F.cross_entropy(x, x2))
#     print(torch.allclose(y, y2))


def binary_cross_entropy_with_logits(pred: Tensor, target: Tensor) -> Tensor:
    """binary_cross_entropy数值不稳定. 
    pred: Tensor[float]. [N]
    target: Tensor[float]. [N]
    """
    # -logsigmoid(x)*target-logsigmoid(-x)*(1-target)
    # logsigmoid(-x)) == log(1 - sigmoid(x))
    ###
    p_sig: Tensor = F.logsigmoid(pred)
    pm_sig: Tensor = F.logsigmoid(-pred)
    res = p_sig.mul_(target)
    res.add_(pm_sig.mul_((1-target)))
    return -res.mean()


# if __name__ == "__main__":
#     x = torch.randn((1000,))
#     x2 = torch.randint(0, 2, (1000,), dtype=torch.float)
#     y = libs_utils.test_time(
#         lambda: binary_cross_entropy_with_logits(x, x2), number=100)
#     y2 = libs_utils.test_time(
#         lambda: F.binary_cross_entropy_with_logits(x, x2), number=100)
#     print(torch.allclose(y, y2))


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: [N, F]. Tensor[float]
    target: [N, F]. Tensor[float]
    """
    # torch.mean((y_pred - y_true) ** 2, dim=0)
    res = target - pred
    res = torch.einsum("ij,ij->", res, res)
    return res.div_(pred.numel())

# if __name__ == "__main__":
#     x = torch.randn((1000, 100))
#     x2 = torch.randn((1000, 100))
#     y = libs_utils.test_time(lambda: mse_loss(x, x2), number=100)
#     y2 = libs_utils.test_time(lambda: F.mse_loss(x, x2), number=100)
#     print(torch.allclose(y, y2))


# if __name__ == "__main__":
#     x = torch.randn((1000, 1000))
#     y = libs_utils.test_time(lambda: x+x+x+x+x+x, number=10)
#     y = libs_utils.test_time(lambda: x+x+x.add_(x).add_(x).add_(x), number=10)
#     y2 = libs_utils.test_time(lambda: x.add_(x).add_(
#         x).add_(x).add_(x).add_(x), number=10)
#     # time[number=10]: 0.000868±0.000660 |max: 0.002473 |min: 0.000315
#     # time[number=10]: 0.000246±0.000100 |max: 0.000496 |min: 0.000177
#     # time[number=10]: 0.000137±0.000058 |max: 0.000308 |min: 0.000107
