# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from typing import List, Tuple, Any, Dict
from torch.nn import Module
from thop import profile
from torch import Tensor
import torch.nn.functional as F
import torch

__all__ = ["freeze_layers", "print_model_info",
           "label_smoothing_cross_entropy", "fuse_conv_bn", "fuse_linear_bn"]


def freeze_layers(model: Module, layer_prefix_names: List[str]) -> None:
    """inplace"""
    for n, p in model.named_parameters():
        requires_grad = True
        for lpn in layer_prefix_names:
            if n.startswith(lpn):
                requires_grad = False
                break
        p.requires_grad_(requires_grad)


def print_model_info(model: Module, inputs: Tuple[Any, ...]) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_grads = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # FLOPs
    macs, _ = profile(model, inputs, verbose=False)
    flops = macs * 2
    #
    n_params /= 1e6
    n_grads /= 1e6
    flops /= 1e9
    s = [
        f"{model.__class__.__name__}: ",
        f"{len(list(model.modules()))} layers, ",
        f"{n_params:.4f}M parameters, ",
        f"{n_grads:.4f}M grads, ",
        f"{flops:.4f}G FLOPs"
    ]
    print("".join(s))


# if __name__ == "__main__":
#     from torchvision.models import resnet50
#     import torch

#     model = resnet50()
#     input = torch.randn(1, 3, 224, 224)
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
#     y3 = libs_utils.test_time(lambda: F.cross_entropy(x, x2), number=10)
#     y = libs_utils.test_time(lambda: label_smoothing_cross_entropy(x, x2, smoothing=0.9), number=10, warm_up=5)
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
