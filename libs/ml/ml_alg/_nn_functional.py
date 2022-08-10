# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
# 用于学习, 速度慢于F.
# 没有显示说明, 就不是inplace的

__all__ = []

"""
relu: x=>0保持, x<0变为=0
leaky_relu: x>=0保持, x<0进行线性压缩(*negative_slope). 
sigmoid: 1/(1+e^x)
tanh: (e^x-e^{-x})/(e^x+e^{-x})
softmax: e^x/{归一化}
silu: x*sigmoid(x)
gelu: x*phi(x). phi(x)是高斯分布的CDF{近似于sigmoid(1.702x)}
"""
if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *


def relu(x: Tensor, inplace=False) -> Tensor:
    """
    x: Tensor[float]. [...]
    """
    if not inplace:
        x = x.clone()
    return x.clamp_min_(0)


# if __name__ == "__main__":
#     x = torch.randn(1000, 1000)
#     x1 = x.clone()
#     y = libs_ml.test_time(lambda: relu(x1))
#     x2 = x.clone()
#     y2 = libs_ml.test_time(lambda: relu(x2, inplace=True))
#     x3 = x.clone()
#     y3 = libs_ml.test_time(lambda: F.relu(x3))
#     x4 = x.clone()
#     y4 = libs_ml.test_time(lambda: F.relu(x4, inplace=True))
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
    """
    x: Tensor[long]. [N]
    return: Tensor[long]. [N, n_classes]
    """
    if n_classes == -1:
        n_classes = x.max() + 1
    res = torch.zeros((x.shape[0], n_classes),
                      dtype=torch.long, device=x.device)
    res[torch.arange(x.shape[0]), x] = 1
    return res


# if __name__ == "__main__":
#     x = torch.randint(0, 10, (1000,))  # long
#     # y = libs_ml.test_time(lambda: _one_hot(x), number=10)
#     y2 = libs_ml.test_time(lambda: one_hot(x), number=10)
#     y3 = libs_ml.test_time(lambda: F.one_hot(x), number=10)
#     # print(torch.allclose(y, y2))
#     print(torch.allclose(y2, y3))

def nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: Tensor[float]. [N, F]
    target: Tensor[long]. [N]
    """
    N, n_labels = pred.shape[:2]
    target = F.one_hot(target, n_labels)  # long
    res = pred.mul(target)
    return -res.sum() / N


# if __name__ == "__main__":
#     x = torch.randn((1000, 10))
#     x2 = torch.randint(0, 10, (1000,))
#     y = libs_ml.test_time(lambda: nll_loss(x, x2))
#     y2 = libs_ml.test_time(lambda: F.nll_loss(x, x2))
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
#     y = libs_ml.test_time(lambda: cross_entropy(x, x2))
#     y2 = libs_ml.test_time(lambda: F.cross_entropy(x, x2))
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
#     y = libs_ml.test_time(
#         lambda: binary_cross_entropy_with_logits(x, x2), number=100)
#     y2 = libs_ml.test_time(
#         lambda: F.binary_cross_entropy_with_logits(x, x2), number=100)
#     print(torch.allclose(y, y2))


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: [N, F]. Tensor[float]
    target: [N, F]. Tensor[float]
    """
    # torch.mean((y_pred - y_true) ** 2, dim=0)
    res = target.sub(pred)
    res = torch.einsum("ij,ij->", res, res)
    return res.div_(pred.numel())

# if __name__ == "__main__":
#     x = torch.randn((1000, 100))
#     x2 = torch.randn((1000, 100))
#     y = libs_ml.test_time(lambda: mse_loss(x, x2), number=100)
#     y2 = libs_ml.test_time(lambda: F.mse_loss(x, x2), number=100)
#     print(torch.allclose(y, y2))


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.) -> Tensor:
    """diff=beta为loss1, loss2的分界线."""
    # 在beta处, loss1, loss2的导数值, 值相等. loss(diff=0)=0
    diff = target.sub(pred).abs_()
    cond = diff.lt(beta)
    loss1 = diff.sub(beta / 2)
    loss2 = diff.mul_(diff).mul_(1/2 / beta)
    return loss2.where(cond, loss1).mean()


# if __name__ == "__main__":
#     x = torch.randn(100000)
#     y = torch.randn(100000)
#     y1 = libs_ml.test_time(lambda: F.smooth_l1_loss(
#         x, y, beta=2), number=10, warm_up=1)
#     y2 = libs_ml.test_time(lambda: smooth_l1_loss(x, y, beta=2), number=10)
#     print(y1, y2)


def _bn_1d(
    x: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
):
    """
    x: [N, F]
    running_mean: [F]. inplace
    running_var: [F]. inplace
    weight: [F]
    bias: [F]
    """
    if training is True:
        mean = x.mean(0)  # []
        var = x.var(0, False)
        N = x.shape[0]
        with torch.no_grad():
            var_unbiased = var.mul(N/(N - 1))
            running_mean.mul_(1-momentum).add_(mean.mul(momentum))
            running_var.mul_(1-momentum).add_(var_unbiased.mul_(momentum))
    else:
        mean = running_mean.clone()
        var = running_var.clone()
    # weight * (x - mean)/sqrt(var + eps) + bias
    scale = var.add_(eps).rsqrt_()
    if weight is not None:
        scale.mul_(weight)
    b = mean.mul_(scale).neg_()
    if bias is not None:
        b.add_(bias)
    return x.mul(scale).add_(b)


def _bn_2d(
    x: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
):
    """
    x: [N, C, H, W]
    running_mean: [F]. inplace
    running_var: [F]. inplace
    weight: [F]
    bias: [F]
    """
    if training is True:
        mean = x.mean((0, 2, 3))  # []
        var = x.var((0, 2, 3), False)
        N = x.numel() // x.shape[1]
        with torch.no_grad():
            var_unbiased = var.mul(N/(N-1))
            running_mean.mul_(1-momentum).add_(mean.mul(momentum))
            running_var.mul_(1-momentum).add_(var_unbiased.mul_(momentum))
    else:
        mean = running_mean.clone()
        var = running_var.clone()
    # weight * (x - mean)/sqrt(var + eps) + bias
    mean = mean[None, :, None, None]
    var = var[None, :, None, None]
    scale = var.add_(eps).rsqrt_()
    if weight is not None:
        weight = weight[None, :, None, None]
        scale.mul_(weight)
    b = mean.mul_(scale).neg_()
    if bias is not None:
        bias = bias[None, :, None, None]
        b.add_(bias)
    return x.mul(scale).add_(b)


def batch_norm(
    x: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5
):
    _ndim = x.ndim
    #
    if _ndim == 2:
        res = _bn_1d(x, running_mean, running_var, weight,
                     bias, training, momentum, eps)
    elif _ndim == 4:
        res = _bn_2d(x, running_mean, running_var, weight,
                     bias, training, momentum, eps)
    else:
        raise ValueError(f"x.ndim: {_ndim}")
    return res


# if __name__ == "__main__":
#     libs_ml.seed_everything(42)
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y1 = libs_ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     rm1 = running_mean
#     rv1 = running_var
#     x1 = x
#     libs_ml.seed_everything(42)
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y2 = libs_ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     print(torch.allclose(x, x1, atol=1e-6))
#     #

#     #
#     libs_ml.seed_everything(42)
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y1 = libs_ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     rm1 = running_mean
#     rv1 = running_var
#     x1 = x
#     libs_ml.seed_everything(42)
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y2 = libs_ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     print(torch.allclose(x, x1, atol=1e-6))
#     #
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     y1 = libs_ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     x1 = x
#     rm1 = running_mean
#     rv1 = running_var
#     y2 = libs_ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     print(torch.allclose(x, x1, atol=1e-6))
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     y1 = libs_ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     x1 = x
#     rm1 = running_mean
#     rv1 = running_var
#     y2 = libs_ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     print(torch.allclose(x, x1, atol=1e-6))
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))


def layer_norm(
    x: Tensor,
    normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """
    x: [N, L, F]
    normalized_shape: 前面补1. 
        一般情况下, normalized_shape为 [F], 表示每一个位置, 一个mean/var. 
        normalized_shape需要和weight, bias的shape一致. 
    weight: [F]
    bias: [F]
    """
    # check normalized_shape
    _dim = []
    for ns, i in zip(reversed(normalized_shape), reversed(range(x.ndim))):
        if ns != x.shape[i]:
            raise ValueError(f"x.shape[{i}]: {x.shape[i]}")
        _dim.append(i)
    if weight is not None:
        assert list(weight.shape) == normalized_shape
    #
    mean = x.mean(_dim, keepdim=True)
    var = x.var(_dim, False, keepdim=True)
    scale = var.add_(eps).rsqrt_()
    if weight is not None:
        scale = scale.mul(weight)

    b = mean.mul(scale).neg_()
    if bias is not None:
        b.add_(bias)
    return scale.mul_(x).add_(b)


# if __name__ == "__main__":
#     libs_ml.seed_everything(42)
#     x = torch.randn(10, 50, 100)
#     w = torch.randn(100)
#     b = torch.randn(100)
#     y1 = libs_ml.test_time(lambda: F.layer_norm(x, [100], w, b))
#     y2 = libs_ml.test_time(lambda: layer_norm(x, [100], w, b))
#     print(torch.allclose(y1, y2, atol=1e-6))
#     w = torch.randn(50, 100)
#     b = torch.randn(50, 100)
#     y1 = libs_ml.test_time(lambda: F.layer_norm(x, [50, 100], w, b))
#     y2 = libs_ml.test_time(lambda: layer_norm(x, [50, 100], w, b))
#     print(torch.allclose(y1, y2, atol=1e-6))

def dropout(
    x: Tensor,
    p: float = 0.5,  # drop_p
    training: bool = True,
    inplace: bool = False
) -> Tensor:
    if not training or p == 0.:
        return x  # 同F.dropout
    keep_p = 1 - p
    keep_tensors = torch.bernoulli(torch.full_like(x, keep_p))
    if not inplace:
        x = x.clone()
    x.mul_(keep_tensors)
    x.div_(keep_p)
    return x


# if __name__ == "__main__":
#     x = torch.randn(100, 100, device='cuda')
#     libs_ml.seed_everything(42)
#     y1: Tensor = libs_ml.test_time(lambda: F.dropout(x, 0.9), warm_up=2)
#     libs_ml.seed_everything(42)
#     y2: Tensor = libs_ml.test_time(lambda: dropout(x, 0.9), warm_up=2)
#     print(torch.allclose(y1, y2))  # True
#     print(y1.count_nonzero(), y2.count_nonzero())

# def conv2d(
#     x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
#     stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
#     dilation: int = 1, groups: int = 1
# ) -> Tensor:
#     """
#     x: [N, Cin, Hin, Win]
#     weight: [Cout, Cin//G, KH, KW].
#     bias: [Cout]
#     stride: SH, SW
#     padding: PH, PW
#     """
#     if padding != (0, 0):
#         x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])  # lrtb
#     Hin, Win = x.shape[2:]
#     D, G = dilation, groups
#     KH, KW = weight.shape[2:]
#     KH_D, KW_D = (KH - 1) * D + 1, (KW - 1) * D + 1
#     SH, SW = stride
#     N, Cin = x.shape[:2]
#     Cout = weight.shape[0]
#     assert weight.shape[1] * G == Cin
#     # Out = (In + 2*P − (K-1)*D+1)) // S + 1. (P, D已经在In, K中算进去了)
#     Hout, Wout = (Hin - KH_D) // SH + 1, (Win - KW_D) // SW + 1
#     res = torch.empty((N, Cout, Hout, Wout), device=x.device, dtype=x.dtype)
#     x = x.contiguous().view(N, G, Cin//G, Hin, Win)
#     weight = weight.contiguous().view(G, Cout // G, Cin//G, KH, KW)
#     for i in range(Hout):
#         for j in range(Wout):
#             h_start, w_start = i * SH, j * SW
#             h_pos, w_pos = slice(h_start, (h_start + KH_D), D), \
#                 slice(w_start, (w_start + KW_D), D)
#             # [N, G, Cin//G, KH, KW], [G, Cout//G, Cin//G, KH, KW] -> [N, G, Cout//G] -> [N, Cout]
#             res[:, :, i, j].copy_(torch.einsum(
#                 "abcde,bfcde->abf", x[:, :, :, h_pos, w_pos], weight).contiguous().view(N, Cout))
#     if bias is not None:
#         res.add_(bias[None, :,  None, None])
#     return res

def conv2d(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0),
    dilation: int = 1, groups: int = 1
) -> Tensor:
    """
    x: [N, Cin, Hin, Win]
    weight: [Cout, Cin//G, KH, KW]. 
    bias: [Cout]
    stride: SH, SW
    padding: PH, PW
    """
    if padding != (0, 0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])  # lrtb
    Hin, Win = x.shape[2:]
    D, G = dilation, groups
    KH, KW = weight.shape[2:]
    KH_D, KW_D = (KH - 1) * D + 1, (KW - 1) * D + 1
    SH, SW = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[0]
    assert weight.shape[1] * G == Cin
    # Out = (In + 2*P − (K-1)*D+1)) // S + 1. (P, D已经在In, K中算进去了)
    Hout, Wout = (Hin - KH_D) // SH + 1, (Win - KW_D) // SW + 1
    res = []
    x = x.contiguous().view(N, G, Cin//G, Hin, Win)
    weight = weight.contiguous().view(G, Cout // G, Cin//G, KH, KW)
    for i in range(Hout):
        for j in range(Wout):
            h_start, w_start = i * SH, j * SW
            h_pos, w_pos = slice(h_start, (h_start + KH_D), D), \
                slice(w_start, (w_start + KW_D), D)
            # [N, G, Cin//G, KH, KW], [G, Cout//G, Cin//G, KH, KW] -> [N, G, Cout//G] -> [N, Cout]
            res.append(torch.einsum(
                "abcde,bfcde->abf", x[:, :, :, h_pos, w_pos], weight))
    res = torch.stack(res, dim=-1).view(N, Cout, Hout, Wout)
    if bias is not None:
        res.add_(bias[None, :,  None, None])
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
#     libs_ml.seed_everything(42, gpu_dtm=True)
#     x = torch.randn(64, 128, 112, 112, device="cuda")
#     w = torch.randn(256, 128, 3, 3, device="cuda")
#     b = torch.randn(256, device="cuda")
#     y2 = libs_ml.test_time(lambda: F.conv2d(
#         x, w, b, (1, 1), (1, 1), 2, 1), 10, timer=libs_ml.time_synchronize)
#     y1 = libs_ml.test_time(lambda: conv2d(
#         x, w, b, (1, 1), (1, 1), 2, 1), 10, timer=libs_ml.time_synchronize)
#     print(torch.allclose(y1, y2, atol=1e-3))
    
#     x = torch.randn(64, 128, 112, 112, device="cuda")
#     w = torch.randn(256, 1, 3, 3, device="cuda")
#     b = torch.randn(256, device="cuda")
#     y2 = libs_ml.test_time(lambda: F.conv2d(
#         x, w, b, (1, 1), (1, 1), 2, 128), 10, timer=libs_ml.time_synchronize)
#     y1 = libs_ml.test_time(lambda: conv2d(
#         x, w, b, (1, 1), (1, 1), 2, 128), 10, timer=libs_ml.time_synchronize)
    
#     print(torch.allclose(y1, y2, atol=1e-3))


def conv1d(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: int = 1, padding: int = 0,
    dilation: int = 1, groups: int = 1
) -> Tensor:
    """
    x: [N, Cin, Lin]
    weight: [Cout, Cin//G, KL]. 
    bias: [Cout]
    stride: SL
    padding: PL
    """
    if padding != 0:
        x = F.pad(x, [padding, padding])  # lr
    Lin = x.shape[2]
    D, G = dilation, groups
    KL = weight.shape[2]
    KL_D = (KL - 1) * D + 1
    SL = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[0]
    assert weight.shape[1] * G == Cin
    # Out = (In + 2*P − (K-1)*D+1)) // S + 1. (P, D已经在In, K中算进去了)
    Lout = (Lin - KL_D) // SL + 1
    res = torch.empty((N, Cout, Lout))
    x = x.contiguous().view(N, G, Cin // G, Lin)
    weight = weight.contiguous().view(G, Cout // G, Cin//G, KL)
    for i in range(Lout):
        l_start = i * SL
        l_pos = slice(l_start, (l_start + KL_D), D)
        # [N, G, Cin//G, KL], [G, Cout//G, Cin//G, KL] -> [N, G, Cout//G]
        res[:, :, i].copy_(torch.einsum(
            "abcd,becd->abe", x[:, :, :, l_pos], weight).contiguous().view(N, Cout))
    if bias is not None:
        res.add_(bias[None, :,  None])
    return res


# if __name__ == "__main__":
#     x = torch.randn(32, 128, 32*32)
#     w = torch.randn(256, 128, 8)
#     b = torch.randn(256)
#     y1 = libs_ml.test_time(lambda: conv1d(x, w, b, 1, 1))
#     y2 = libs_ml.test_time(lambda: F.conv1d(x, w, b, 1, 1))
#     print(torch.allclose(y1, y2, atol=1e-4))
#     #
#     x = torch.randn(32, 128, 32*32)
#     w = torch.randn(256, 1, 7)
#     b = torch.randn(256)
#     y1 = libs_ml.test_time(lambda: conv1d(x, w, b, 1, 1, 2, 128))
#     y2 = libs_ml.test_time(lambda: F.conv1d(x, w, b, 1, 1, 2, 128))
#     print(torch.allclose(y1, y2, atol=1e-4))


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    x: [N, F]
    weight: [F2, F]
    bias: [F2]
    """
    res = torch.einsum("ab,cb->ac", x, weight)
    if bias is not None:
        res.add_(bias)
    return res

# if __name__ == "__main__":
#     x = torch.randn(100, 128)
#     w = torch.randn(256, 128)
#     b = torch.randn(256)
#     libs_ml.test_time(lambda:linear(x, w, b), number=10)
#     libs_ml.test_time(lambda:F.linear(x, w, b), number=10)


def lstm_cell(
        x: Tensor, hx: Tuple[Tensor, Tensor],
        w_ih: Tensor, w_hh: Tensor,
        b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """
    x: [N, Cin]
    hx: Tuple[h, c], h: [N, Ch], c: [N, Ch]
    w_ih: [4*Ch, Cin]. i,f,g,o
    w_hh: [4*Ch, Ch].
    b_ih: [4*Ch].
    b_hh: [4*Ch].
    return: Tuple[y, c_], y: [N, Ch], c: [N, Ch]. y也可以理解为h_
    """

    Cin = x.shape[1]
    Ch = w_ih.shape[0] // 4
    h, c = hx
    w_ih = w_ih.contiguous().view(4, Ch, Cin)
    w_hh = w_hh.contiguous().view(4, Ch, Ch)
    if b_ih is not None:
        b_ih = b_ih.contiguous().view(4, Ch)
    else:
        b_ih = (None, None, None, None)
    if b_hh is not None:
        b_hh = b_hh.contiguous().view(4, Ch)
    else:
        b_hh = (None, None, None, None)
    #
    i = (F.linear(x, w_ih[0], b_ih[0]) +
         F.linear(h, w_hh[0], b_hh[0])).sigmoid_()  # 输入门
    f = (F.linear(x, w_ih[1], b_ih[1]) +
         F.linear(h, w_hh[1], b_hh[1])).sigmoid_()  # 遗忘门
    g = (F.linear(x, w_ih[2], b_ih[2]) +
         F.linear(h, w_hh[2], b_hh[2])).tanh_()  # 输入信息
    o = (F.linear(x, w_ih[3], b_ih[3]) +
         F.linear(h, w_hh[3], b_hh[3])).sigmoid_()  # 输出门
    # 可以看到c会受到梯度消失的影响(f门).
    c_ = f.mul_(c).add_(i.mul_(g))  # c_ = f * c + i * g
    y = o.mul_(c_.tanh())  # y = o * tanh(c_)  # 对c信息化
    return y, c_


# if __name__ == "__main__":
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256), torch.randn(100, 256)
#     w_ih = torch.randn(4 * 256, 128)
#     w_hh = torch.randn(4 * 256, 256)
#     b_ih = torch.randn(4 * 256)
#     b_hh = torch.randn(4 * 256)
#     y1 = libs_ml.test_time(lambda: torch.lstm_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     y2 = libs_ml.test_time(lambda: lstm_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))
#     #
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256), torch.randn(100, 256)
#     w_ih = torch.randn(4 * 256, 128)
#     w_hh = torch.randn(4 * 256, 256)
#     y1 = libs_ml.test_time(lambda: torch.lstm_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     y2 = libs_ml.test_time(lambda: lstm_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))


def gru_cell(
        x: Tensor, hx: Tensor,
        w_ih: Tensor, w_hh: Tensor,
        b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None) -> Tensor:
    """
    x: [N, Cin]
    hx: [N, Ch]
    w_ih: [3*Ch, Cin]. r,z,n
    w_hh: [3*Ch, Ch].
    b_ih: [3*Ch].
    b_hh: [3*Ch].
    return: y. y也可以理解为hx_
    """

    Cin = x.shape[1]
    Ch = w_ih.shape[0] // 3
    w_ih = w_ih.contiguous().view(3, Ch, Cin)
    w_hh = w_hh.contiguous().view(3, Ch, Ch)
    if b_ih is not None:
        b_ih = b_ih.contiguous().view(3, Ch)
    else:
        b_ih = (None, None, None)
    if b_hh is not None:
        b_hh = b_hh.contiguous().view(3, Ch)
    else:
        b_hh = (None, None, None)
    #
    r = (F.linear(x, w_ih[0], b_ih[0]) +
         F.linear(hx, w_hh[0], b_hh[0])).sigmoid_()  # 重置门
    z = (F.linear(x, w_ih[1], b_ih[1]) +
         F.linear(hx, w_hh[1], b_hh[1])).sigmoid_()  # 更新门
    n = (F.linear(x, w_ih[2], b_ih[2]) +
         r.mul_(F.linear(hx, w_hh[2], b_hh[2]))).tanh_()  # 输入信息
    # 可以看到hx会受到梯度消失的影响(z门).
    y = (z.neg().add_(1)).mul_(n).add_(z.mul_(hx))  # (1 - z) * n + z * hx
    return y

# if __name__ == "__main__":
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256)
#     w_ih = torch.randn(3 * 256, 128)
#     w_hh = torch.randn(3 * 256, 256)
#     b_ih = torch.randn(3 * 256)
#     b_hh = torch.randn(3 * 256)
#     y1 = libs_ml.test_time(lambda: torch.gru_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     y2 = libs_ml.test_time(lambda: gru_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))
#     #
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256)
#     w_ih = torch.randn(3 * 256, 128)
#     w_hh = torch.randn(3 * 256, 256)
#     y1 = libs_ml.test_time(lambda: torch.gru_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     y2 = libs_ml.test_time(lambda: gru_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))


def _scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    training: bool = True
) -> Tuple[Tensor, Tensor]:
    """Attention + dropout
    Q: [T, N*H, E//H]
    K: [S, N*H, E//H]
    V: [S, N*H, E//H]
    attn_mask: [N*H, T, S]
    return: res, W
        res: [T, N*H, E//H]
        W: [N*H, T, S]
    """
    E_DIV_H = Q.shape[2]
    # [T, N*H, E//H], [S, N*H, E//H] -> [N*H, T, S]
    W = torch.einsum("abc,dbc->bad", Q, K).div_(math.sqrt(E_DIV_H))
    if attn_mask is not None:
        W.add_(attn_mask)
    W = W.softmax(dim=-1)
    if dropout_p > 0.:
        F.dropout(W, dropout_p, training, inplace=True)
    # [N*H, T, S], [S, N*H, E//H] -> [T, N*H, E//H]
    res = torch.einsum("abc,cad->bad", W, V)
    return res, W


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    # embed_dim_to_check: int,  # 对E的check, 就是embed_dim
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    # bias_k: Optional[Tensor],  # None
    # bias_v: Optional[Tensor],  # None
    # add_zero_attn: bool,  # False
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    # use_separate_proj_weight: bool = False,  # 当dk!=dv时使用. 此情况为少数
    # q_proj_weight: Optional[Tensor] = None,
    # k_proj_weight: Optional[Tensor] = None,
    # v_proj_weight: Optional[Tensor] = None,
    # static_k: Optional[Tensor] = None,
    # static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,  # 对参数进行平均(若need_weights)

) -> Tuple[Tensor, Optional[Tensor]]:
    """
    query: [T, N, E]. 上层Q
    key: [S, N, E]. 下层的K, V
    value: [S, N, E]
    in_proj_weight: [3E, E]
    in_proj_bias: [3E]
    out_proj_weight: [E, E]
    out_proj_bias: [E]
    key_padding_mask: [N, S]. Tensor[bool], True代表mask(同masked_fill). 对[PAD]进行mask
    attn_mask: [T, S] or [N*H, T, S]. Tensor[bool], True代表mask. 对因果进行mask
    return: output: [T, N, E], weights: [N, T, S] or [N, H, T, S]
    """
    # mask(-inf), 前线性映射, multi-head, Attention, mask, 后线性映射
    T, N, E = query.shape
    S = key.shape[0]
    H = num_heads
    Q, K, V = query, key, value
    # mask. [N, S], [T, S] -> [N*H, T, S]
    if attn_mask is not None:
        if attn_mask.ndim == 2:
            attn_mask = attn_mask[None]  # [1, T, S]
    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.contiguous().view(
            N, 1, 1, S).expand(N, H, 1, S).contiguous().view(N*H, 1, S)
        if attn_mask is not None:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = key_padding_mask
    if attn_mask is not None:
        new_mask = torch.zeros_like(attn_mask, dtype=Q.dtype)
        new_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_mask

    #
    in_proj_weight = in_proj_weight.contiguous().view(3, E, E)
    if in_proj_bias is not None:
        in_proj_bias = in_proj_bias.contiguous().view(3, E)
    else:
        in_proj_bias = (None, None, None)
    # [T, N, E], [E, E] -> [T, N, E] -> [T, N*H, E//H]
    Q = F.linear(Q, in_proj_weight[0], in_proj_bias[0]).view(T, N*H, E//H)
    K = F.linear(K, in_proj_weight[1], in_proj_bias[1]).view(S, N*H, E//H)
    V = F.linear(V, in_proj_weight[2], in_proj_bias[2]).view(S, N*H, E//H)

    # res: [T, N*H, E//H], W: [N*H, T, S]
    res, W = _scaled_dot_product_attention(
        Q, K, V, attn_mask, dropout_p, training)
    #
    res = res.contiguous().view(T, N, E)
    W = W.contiguous().view(N, H, T, S)
    res = F.linear(res, out_proj_weight, out_proj_bias)
    #
    if not need_weights:
        W = None
    elif average_attn_weights:  # need_weights
        # [N, H, T, S] -> [N, T, S]
        W = W.mean(dim=1)
    return res, W


# if __name__ == "__main__":
#     T, N, E = 512, 16, 512
#     S = 256
#     libs_ml.seed_everything(42)
#     query = torch.randn(T, N, E)
#     key = torch.randn(S, N, E)
#     value = torch.randn(S, N, E)
#     embed_dim_to_check = E
#     in_proj_weight = torch.randn(3*E, E)
#     in_proj_bias = torch.randn(3*E)
#     out_proj_weight = torch.randn(E, E)
#     out_proj_bias = torch.randn(E)
#     key_padding_mask = torch.randint(0, 2, (N, S), dtype=torch.bool)
#     attn_mask = torch.randint(0, 2, (T, S), dtype=torch.bool)
#     num_heads = 8

#     libs_ml.seed_everything(42)
#     y1 = libs_ml.test_time(lambda: multi_head_attention_forward(
#         query, key, value, num_heads,
#         in_proj_weight, in_proj_bias, 0.1,
#         out_proj_weight, out_proj_bias, True,
#         key_padding_mask, True, attn_mask), number=10, warm_up=1)
#     libs_ml.seed_everything(42)
#     y2 = libs_ml.test_time(lambda: F.multi_head_attention_forward(
#         query, key, value, embed_dim_to_check, num_heads,
#         in_proj_weight, in_proj_bias, None, None, False, 0.1,
#         out_proj_weight, out_proj_bias, True,
#         key_padding_mask, True, attn_mask), number=10, warm_up=1)
#     print(torch.allclose(y1[0], y2[0], atol=1e-6))
#     print(torch.allclose(y1[1], y2[1], atol=1e-6))
