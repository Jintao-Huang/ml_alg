# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:
from ...._types import *
# from libs import *


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
    return: [N, F]
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
    x: [N, C=F, H, W]
    running_mean: [F]. inplace
    running_var: [F]. inplace
    weight: [F]
    bias: [F]
    return: [N, C=F, H, W]
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
#     ml.seed_everything(42)
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y1 = ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     rm1 = running_mean
#     rv1 = running_var
#     x1 = x
#     ml.seed_everything(42)
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y2 = ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     print(torch.allclose(x, x1, atol=1e-6))
#     #

#     #
#     ml.seed_everything(42)
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y1 = ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     rm1 = running_mean
#     rv1 = running_var
#     x1 = x
#     ml.seed_everything(42)
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     weight = torch.randn(100)
#     bias = torch.randn(100)
#     y2 = ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5), number=2)
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     print(torch.allclose(x, x1, atol=1e-6))
#     #
#     x = torch.randn(1000, 100)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     y1 = ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     x1 = x
#     rm1 = running_mean
#     rv1 = running_var
#     y2 = ml.test_time(lambda:
#                               batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     print(torch.allclose(x, x1, atol=1e-6))
#     print(torch.allclose(y1, y2, atol=1e-6))
#     print(torch.allclose(rm1, running_mean, atol=1e-6))
#     print(torch.allclose(rv1, running_var, atol=1e-6))
#     x = torch.randn(1000, 100, 10, 10)
#     running_mean = torch.randn(100)
#     running_var = torch.randn(100).abs_()
#     y1 = ml.test_time(lambda:
#                               F.batch_norm(x, running_mean, running_var, None, None, False, 0.1, 1e-5), number=2)
#     x1 = x
#     rm1 = running_mean
#     rv1 = running_var
#     y2 = ml.test_time(lambda:
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
    weight: [F]; 则mean/std: [N, L]
    bias: [F]
    return: [N, L, F]
    """
    # check normalized_shape
    _dim = []
    for i, ns in enumerate(normalized_shape, start=x.ndim-len(normalized_shape)):
        assert ns == x.shape[i]
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
#     ml.seed_everything(42)
#     x = torch.randn(10, 50, 100)
#     w = torch.randn(100)
#     b = torch.randn(100)
#     y1 = ml.test_time(lambda: F.layer_norm(x, [100], w, b))
#     y2 = ml.test_time(lambda: layer_norm(x, [100], w, b))
#     print(torch.allclose(y1, y2, atol=1e-6))
#     w = torch.randn(50, 100)
#     b = torch.randn(50, 100)
#     y1 = ml.test_time(lambda: F.layer_norm(x, [50, 100], w, b))
#     y2 = ml.test_time(lambda: layer_norm(x, [50, 100], w, b))
#     print(torch.allclose(y1, y2, atol=1e-6))

def dropout(
    x: Tensor,
    p: float = 0.5,  # drop_p
    training: bool = True,
    inplace: bool = False
) -> Tensor:
    # 和F.dropout的结果不同
    if not training or p == 0.:
        return x  # 同F.dropout
    if not inplace:
        x = x.clone()
    x.mul_(torch.rand_like(x) > p)
    x.div_(1-p)
    return x


# if __name__ == "__main__":
#     x = torch.ones(100, 100, device='cuda')
#     ml.seed_everything(42)
#     y1: Tensor = ml.test_time(lambda: F.dropout(x, 0.9), warmup=2)
#     ml.seed_everything(42)
#     y2: Tensor = ml.test_time(lambda: dropout(x, 0.9), warmup=2)
#     print(torch.allclose(y1, y2))  # False
#     print(y1.count_nonzero(), y2.count_nonzero())


def dropout1d(
    x: Tensor,
    p: float = 0.5,  # drop_p
    training: bool = True,
    inplace: bool = False
) -> Tensor:
    # dropout1d, 与dropout使用相同的随机方法, 这里使用dropout实现.
    if not training or p == 0.:
        return x
    if not inplace:
        x = x.clone()
    mask_shape = x.shape[:-1]
    mask = torch.full(mask_shape, (1-p), dtype=x.dtype, device=x.device)
    F.dropout(mask, p, True, inplace=True)
    x.mul_(mask[..., None])
    x.div_(1-p)
    return x


# if __name__ == "__main__":
#     x = torch.ones((100, 100), device='cuda')
#     ml.seed_everything(42)
#     y1: Tensor = ml.test_time(lambda: F.dropout1d(x, 0.9), warmup=2)
#     ml.seed_everything(42)
#     y2: Tensor = ml.test_time(lambda: dropout1d(x, 0.9), warmup=2)
#     print(torch.allclose(y1, y2))


def dropout2d(
    x: Tensor,
    p: float = 0.5,  # drop_p
    training: bool = True,
    inplace: bool = False
) -> Tensor:
    # dropout2d, 与dropout使用相同的随机方法, 这里使用dropout实现.
    if not training or p == 0.:
        return x
    if not inplace:
        x = x.clone()
    mask_shape = x.shape[:-2]
    mask = torch.full(mask_shape, (1-p), dtype=x.dtype, device=x.device)
    F.dropout(mask, p, True, inplace=True)
    x.mul_(mask[..., None, None])
    x.div_(1-p)
    return x


# if __name__ == "__main__":
#     x = torch.ones((100, 100, 100, 100), device='cuda')
#     ml.seed_everything(42)
#     y1: Tensor = ml.test_time(lambda: F.dropout2d(x, 0.9), warmup=2)
#     ml.seed_everything(42)
#     y2: Tensor = ml.test_time(lambda: dropout2d(x, 0.9), warmup=2)
#     print(torch.allclose(y1, y2))

def conv2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1
) -> Tensor:
    """faster than conv2d_2, but more memory. (recommend)
    x: [N, Cin, Hin, Win]
    weight: [Cout, Cin//G, KH, KW]. 
    bias: [Cout]
    stride: SH, SW
    padding: PH, PW
    return: [N, Cout, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    DH, DW = dilation
    G = groups
    KH, KW = weight.shape[2:]
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    PH, PW = padding
    SH, SW = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[0]
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1
    Hout, Wout = (Hin + 2 * PH - KH_D) // SH + 1, (Win + 2 * PW - KW_D) // SW + 1
    assert weight.shape[1] * G == Cin
    assert Cout % G == 0
    # [N, Cin, Hin, Win] -> [N, Cin*KH*KW, Hout*Wout] -> [N, G, Cin//G, KH*KW, Hout*Wout]
    x = F.unfold(x, (KH, KW), (DH, DW), (PH, PW), (SH, SW))
    x = x.view(N, G, Cin//G, KH*KW, Hout*Wout)
    #
    weight = weight.contiguous().view(G, Cout // G, Cin//G, KH*KW)
    # [N, G, Cin//G, KH*KW, Hout*Wout], [G, Cout//G, Cin//G, KH*KW] ->
    #   [N, G, Cout//G, Hout*Wout] -> [N, Cout, Hout, Wout]
    res = torch.einsum("abcde,bfcd->abfe", x, weight).contiguous().view(N, Cout, Hout, Wout)
    #
    if bias is not None:
        res.add_(bias[None, :,  None, None])
    return res


def conv2d_2(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
    groups: int = 1
) -> Tensor:
    """
    x: [N, Cin, Hin, Win]
    weight: [Cout, Cin//G, KH, KW]. 
    bias: [Cout]
    stride: SH, SW
    padding: PH, PW
    return: [N, Cout, Hout, Wout]
    计算复杂度: O(N Cout Cin Hout Wout KH KW // G)
    """
    if padding != (0, 0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])  # lrtb
    Hin, Win = x.shape[2:]
    DH, DW = dilation
    G = groups
    KH, KW = weight.shape[2:]
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    SH, SW = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[0]
    assert weight.shape[1] * G == Cin
    assert Cout % G == 0
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1. (P, D已经在In, K中算进去了)
    Hout, Wout = (Hin - KH_D) // SH + 1, (Win - KW_D) // SW + 1
    x = x.contiguous().view(N, G, Cin//G, Hin, Win)
    weight = weight.contiguous().view(G, Cout // G, Cin//G, KH, KW)
    res = []
    for i in range(Hout):
        for j in range(Wout):
            h_start, w_start = i * SH, j * SW
            h_pos, w_pos = slice(h_start, (h_start + KH_D), DH), \
                slice(w_start, (w_start + KW_D), DW)
            # [N, G, Cin//G, KH, KW], [G, Cout//G, Cin//G, KH, KW] -> [N, G, Cout//G] -> [N, Cout]
            res.append(torch.einsum("abcde,bfcde->abf", x[:, :, :, h_pos, w_pos], weight))
    res = torch.stack(res, dim=-1).view(N, Cout, Hout, Wout)
    if bias is not None:
        res.add_(bias[None, :,  None, None])
    return res


# if __name__ == "__main__":
#     ml.seed_everything(42, gpu_dtm=True)
#     x = torch.randn(16, 128, 112, 112, device="cuda")
#     w = torch.randn(256, 128, 3, 3, device="cuda")
#     b = torch.randn(256, device="cuda")
#     y1 = ml.test_time(lambda: F.conv2d(
#         x, w, b, (1, 1), (1, 1), (2, 2), 1), 10, timer=ml.time_synchronize)
#     y2 = ml.test_time(lambda: conv2d(
#         x, w, b, (1, 1), (1, 1), (2, 2), 1), 10, timer=ml.time_synchronize)
#     y3 = ml.test_time(lambda: conv2d_2(
#         x, w, b, (1, 1), (1, 1), (2, 2), 1), 10, timer=ml.time_synchronize)
#     print(torch.allclose(y1, y2, atol=1e-3))
#     print(torch.allclose(y2, y3, atol=1e-3))

#     x = torch.randn(16, 128, 112, 112, device="cuda")
#     w = torch.randn(256, 1, 3, 3, device="cuda")
#     b = torch.randn(256, device="cuda")
#     y1 = ml.test_time(lambda: F.conv2d(
#         x, w, b, (1, 1), (1, 1), (2, 2), 128), 10, timer=ml.time_synchronize)
#     y2 = ml.test_time(lambda: conv2d(
#         x, w, b, (1, 1), (1, 1), (2, 2), 128), 10, timer=ml.time_synchronize)
#     y3 = ml.test_time(lambda: conv2d_2(
#         x, w, b, (1, 1), (1, 1), (2, 2), 128), 10, timer=ml.time_synchronize)
#     print(torch.allclose(y1, y2, atol=1e-3))
#     print(torch.allclose(y1, y3, atol=1e-3))


def conv_transpose2d(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_padding: Tuple[int, int] = (0, 0),
    groups: int = 1,
    dilation: Tuple[int, int] = (1, 1)
) -> Tensor:
    """faster than conv_transpose2d_2
    x: [N, Cin, Hin, Win]
    weight: [Cin, Cout//G, KH, KW]. 
    bias: [Cout]
    stride: SH, SW
    padding: PH, PW
    output_padding: OPH, OPW
    return: [N, Cout, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    DH, DW = dilation
    G = groups
    KH, KW = weight.shape[2:]
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    PH, PW = padding
    OPH, OPW = output_padding
    SH, SW = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[1] * G
    # Out = (In - 1) * S - 2*P + (K-1)*D -1 + OP
    Hout, Wout = (Hin - 1) * SH - 2 * PH + KH_D + OPH, (Win - 1) * SW - 2 * PW + KW_D + OPW
    assert Cin % G == 0
    # [N, Cin, Hin, Win] -> [N, G, Cin//G, Hin*Win]
    x = x.view(N, G, Cin//G, Hin*Win)
    #
    weight = weight.contiguous().view(G, Cin // G, Cout//G, KH*KW)
    # [N, G, Cin//G, Hin*Win], [G, Cin//G, Cout//G, KH*KW] ->
    #   [N, G, Cout//G, KH*KW, Hin*Win] -> [N, Cout*KH*KW, Hin*Win]
    res = torch.einsum("abcd,bcfe->abfed", x, weight).contiguous().view(N, Cout*KH*KW, Hin*Win)
    # [N, Cout*KH*KW, Hin*Win] -> [N, Cout, Hout, Wout]
    res = F.fold(res, (Hout, Wout), (KH, KW), (DH, DW), (PH, PW), (SH, SW))
    #
    if bias is not None:
        res.add_(bias[None, :,  None, None])
    return res


# if __name__ == "__main__":
#     x = torch.randn(128, 32, 112, 112)
#     w = torch.randn(32, 16, 3, 3)
#     b = torch.randn(16)
#     y1 = ml.test_time(lambda: F.conv_transpose2d(x, w, b))
#     y2 = ml.test_time(lambda: conv_transpose2d(x, w, b))
#     print(torch.allclose(y1, y2))
#     #
#     x = torch.randn(128, 32, 112, 112)
#     w = torch.randn(32, 4, 3, 3)
#     b = torch.randn(16)
#     y1 = ml.test_time(lambda: F.conv_transpose2d(x, w, b, (2, 2), (1, 1), (1, 1), 4, (2, 2)))
#     y2 = ml.test_time(lambda: conv_transpose2d(x, w, b, (2, 2), (1, 1), (1, 1), 4, (2, 2)))
#     print(torch.allclose(y1, y2))


def conv_transpose2d_2(
    x: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    output_padding: Tuple[int, int] = (0, 0),
    groups: int = 1,
    dilation: Tuple[int, int] = (1, 1)
) -> Tensor:
    """
    x: [N, Cin, Hin, Win]
    weight: [Cin, Cout//G, KH, KW]. 
    bias: [Cout]
    stride: SH, SW
    padding: PH, PW
    output_padding: OPH, OPW. OPH只填充bottom, OPW只填充right(单边).
    return: [N, Cout, Hout, Wout]
    计算复杂度: O(N Cout Cin Hin Win KH KW // G)
    """
    Hin, Win = x.shape[2:]
    DH, DW = dilation
    G = groups
    KH, KW = weight.shape[2:]
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    PH, PW = padding
    OPH, OPW = output_padding
    SH, SW = stride
    N, Cin = x.shape[:2]
    Cout = weight.shape[1] * G
    # Out = (In - 1) * S - 2*P + (K-1)*D -1 + OP
    Hout, Wout = (Hin - 1) * SH - 2 * PH + KH_D + OPH, (Win - 1) * SW - 2 * PW + KW_D + OPW
    assert Cin % G == 0
    x = x.view(N, G, Cin//G, Hin, Win)
    weight = weight.contiguous().view(G, Cin // G, Cout//G, KH, KW)
    #
    res = torch.zeros((N, Cout, Hout + 2 * PH, Wout + 2 * PW))
    for i in range(Hin):
        for j in range(Win):
            h_start, w_start = i * SH, j * SW
            h_pos, w_pos = slice(h_start, (h_start + KH_D), DH), \
                slice(w_start, (w_start + KW_D), DW)
            # [N, G, Cin//G, KH, KW], [G, Cin//G, Cout//G]
            #   -> [N, G, Cout//G, KH, KW] -> [N, Cout, KH, KW]
            res[:, :, h_pos, w_pos].add_(
                torch.einsum("abc,bcfde->abfde", x[:, :, :, i, j], weight).contiguous().view(N, Cout, KH, KW))
    if bias is not None:
        res.add_(bias[None, :,  None, None])
    if PH > 0:
        res = res[:, :, PH:-PH]
    if PW > 0:
        res = res[:, :, :, PW:-PW]
    return res


# if __name__ == "__main__":
#     x = torch.randn(128, 32, 112, 112)
#     w = torch.randn(32, 16, 3, 3)
#     b = torch.randn(16)
#     y1 = ml.test_time(lambda: F.conv_transpose2d(x, w, b))
#     y2 = ml.test_time(lambda: conv_transpose2d_2(x, w, b))
#     print(torch.allclose(y1, y2, atol=1e-5))

#     x = torch.randn(128, 32, 112, 112)
#     w = torch.randn(32, 4, 7, 7)
#     b = torch.zeros(16)
#     y1 = ml.test_time(lambda: F.conv_transpose2d(x, w, b, (5, 5), (1, 1), (3, 4), 4, (1, 1)))
#     y2 = ml.test_time(lambda: conv_transpose2d_2(x, w, b, (5, 5), (1, 1), (3, 4), 4, (1, 1)))
#     print(torch.allclose(y1, y2, atol=1e-5))


def conv1d(
    x: Tensor, weight: Tensor, bias: Optional[Tensor] = None,
    stride: int = 1, padding: int = 0,
    dilation: int = 1, groups: int = 1
) -> Tensor:
    """faster
    x: [N, Cin, Lin]
    weight: [Cout, Cin//G, KL]. 
    bias: [Cout]
    stride: SL
    padding: PL
    return: [N, Cout, Lout]
    """
    Lin = x.shape[2]
    S, P, D, G = stride, padding, dilation, groups
    K = weight.shape[2]
    K_D = (K - 1) * D + 1
    N, Cin = x.shape[:2]
    Cout = weight.shape[0]
    assert weight.shape[1] * G == Cin
    # Out = (In + 2*P − (K-1)*D+1)) // S + 1. (P, D已经在In, K中算进去了)
    Lout = (Lin + 2*P - K_D) // S + 1
    x = F.unfold(x[..., None], (K, 1), D, (P, 0), (S, 1))
    x = x.view(N, G, Cin // G, K, Lout)
    weight = weight.contiguous().view(G, Cout // G, Cin//G, K)
    # x: [N, G, Cout//G, Lout] -> [N, Cout, Lout]
    res = torch.einsum("abcde,bfcd->abfe", x, weight).contiguous().view(N, Cout, Lout)
    if bias is not None:
        res.add_(bias[None, :,  None])
    return res


def conv1d_2(
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
    return: [N, Cout, Lout]
    计算复杂度: O(N Cout Cin Lout KL // G)
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
    x = x.contiguous().view(N, G, Cin // G, Lin)
    weight = weight.contiguous().view(G, Cout // G, Cin//G, KL)
    res = []
    for i in range(Lout):
        l_start = i * SL
        l_pos = slice(l_start, (l_start + KL_D), D)
        # [N, G, Cin//G, KL], [G, Cout//G, Cin//G, KL] -> [N, G, Cout//G]
        res.append(torch.einsum(
            "abcd,becd->abe", x[:, :, :, l_pos], weight).contiguous().view(N, Cout))
    res = torch.stack(res, dim=-1).view(N, Cout, Lout)
    if bias is not None:
        res.add_(bias[None, :,  None])
    return res


# if __name__ == "__main__":
#     x = torch.randn(32, 128, 32*32)
#     w = torch.randn(256, 128, 8)
#     b = torch.randn(256)
#     y1 = ml.test_time(lambda: F.conv1d(x, w, b, 1, 1))
#     y2 = ml.test_time(lambda: conv1d(x, w, b, 1, 1))
#     y3 = ml.test_time(lambda: conv1d_2(x, w, b, 1, 1))
#     print(torch.allclose(y1, y2, atol=1e-4))
#     print(torch.allclose(y1, y3, atol=1e-4))
#     #
#     x = torch.randn(32, 128, 32*32)
#     w = torch.randn(256, 1, 7)
#     b = torch.randn(256)
#     y1 = ml.test_time(lambda: F.conv1d(x, w, b, 1, 1, 2, 128))
#     y2 = ml.test_time(lambda: conv1d(x, w, b, 1, 1, 2, 128))
#     y3 = ml.test_time(lambda: conv1d_2(x, w, b, 1, 1, 2, 128))
#     print(torch.allclose(y1, y2, atol=1e-4))
#     print(torch.allclose(y1, y3, atol=1e-4))


def avg_pool2d(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding: Tuple[int, int] = (0, 0),
) -> Tensor:
    """
    x: [N, C, Hin, Win]
    kernel_size: KH, KW
    stride: SH, SW. None则和kernel_size一致
    padding: PH, PW
    return: [N, C, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    KH, KW = kernel_size
    PH, PW = padding
    if stride is None:
        stride = kernel_size
    SH, SW = stride
    N, C = x.shape[:2]
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1
    Hout, Wout = (Hin + 2 * PH - KH) // SH + 1, (Win + 2 * PW - KW) // SW + 1
    # [N, C, Hin, Win] -> [N, C*KH*KW, Hout*Wout] -> [N*C, KH*KW, Hout*Wout]
    x = F.unfold(x, (KH, KW), 1, (PH, PW), (SH, SW))
    x = x.view(N, C, KH*KW, Hout, Wout)
    res = x.mean(dim=2)
    return res


def avg_pool2d_2(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding: Tuple[int, int] = (0, 0),
) -> Tensor:
    """
    x: [N, C, Hin, Win]
    kernel_size: KH, KW
    stride: SH, SW. None则和kernel_size一致
    padding: PH, PW
    return: [N, C, Hout, Wout]
    """
    if padding != (0, 0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])  # lrtb
    Hin, Win = x.shape[2:]
    KH, KW = kernel_size
    if stride is None:
        stride = kernel_size
    SH, SW = stride
    N, C = x.shape[:2]
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1
    Hout, Wout = (Hin - KH) // SH + 1, (Win - KW) // SW + 1
    res = []
    for i in range(Hout):
        for j in range(Wout):
            h_start, w_start = i * SH, j * SW
            h_pos, w_pos = slice(h_start, (h_start + KH)), \
                slice(w_start, (w_start + KW))
            res.append(torch.mean(x[:, :, h_pos, w_pos], dim=(2, 3)))
    res = torch.stack(res, dim=-1).view(N, C, Hout, Wout)
    return res


def max_pool2d(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tensor:
    """
    x: [N, C, Hin, Win]
    kernel_size: KH, KW
    stride: SH, SW. None则和kernel_size一致
    padding: PH, PW
    return: [N, C, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    DH, DW = dilation
    KH, KW = kernel_size
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    PH, PW = padding
    if stride is None:
        stride = kernel_size
    SH, SW = stride
    N, C = x.shape[:2]
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1
    Hout, Wout = (Hin + 2 * PH - KH_D) // SH + 1, (Win + 2 * PW - KW_D) // SW + 1
    # [N, C, Hin, Win] -> [N, C*KH*KW, Hout*Wout] -> [N*C, KH*KW, Hout*Wout]
    if padding != (0, 0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]], value=float("-inf"))  # lrtb
    x = F.unfold(x, (KH, KW), (DH, DW), (0, 0), (SH, SW))
    x = x.view(N, C, KH*KW, Hout, Wout)
    res = x.max(dim=2)[0]
    return res


def max_pool2d_2(
    x: Tensor,
    kernel_size: Tuple[int, int],
    stride: Optional[Tuple[int, int]] = None,
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1)
) -> Tensor:
    """
    x: [N, C, Hin, Win]
    kernel_size: KH, KW
    stride: SH, SW. None则和kernel_size一致
    padding: PH, PW
    return: [N, C, Hout, Wout]
    """
    if padding != (0, 0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]], value=float("-inf"))  # lrtb
    Hin, Win = x.shape[2:]
    KH, KW = kernel_size
    if stride is None:
        stride = kernel_size
    SH, SW = stride
    DH, DW = dilation
    KH_D, KW_D = (KH - 1) * DH + 1, (KW - 1) * DW + 1
    N, C = x.shape[:2]
    # Out = (In + 2*P − ((K-1)*D+1)) // S + 1
    Hout, Wout = (Hin - KH_D) // SH + 1, (Win - KW_D) // SW + 1
    res = []
    for i in range(Hout):
        for j in range(Wout):
            h_start, w_start = i * SH, j * SW
            h_pos, w_pos = slice(h_start, (h_start + KH_D), DH), \
                slice(w_start, (w_start + KW_D), DW)
            res.append(torch.max(x[:, :, h_pos, w_pos].flatten(2, 3), dim=2)[0])
    res = torch.stack(res, dim=-1).view(N, C, Hout, Wout)
    return res


# if __name__ == "__main__":
#     x = torch.randn(128, 32, 224, 224)
#     y = ml.test_time(lambda: F.avg_pool2d(x, (2, 2), (3, 3), (1, 1)), 3)
#     y2 = ml.test_time(lambda: avg_pool2d(x, (2, 2), (3, 3), (1, 1)), 3)
#     y3 = ml.test_time(lambda: avg_pool2d_2(x, (2, 2), (3, 3), (1, 1)), 3)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, y3, atol=1e-6))
#     #
#     x = torch.randn(128, 32, 224, 224)
#     y = ml.test_time(lambda: F.max_pool2d(x, (2, 2), (3, 3), (1, 1), (2, 2)), 3)
#     y2 = ml.test_time(lambda: max_pool2d(x, (2, 2), (3, 3), (1, 1), (2, 2)), 3)
#     y3 = ml.test_time(lambda: max_pool2d_2(x, (2, 2), (3, 3), (1, 1), (2, 2)), 3)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, y3, atol=1e-6))
#     #
#     x = torch.randn(128, 32, 10, 10)
#     print(F.max_pool2d(x, (3, 3)).shape)  # torch.Size([128, 32, 3, 3])


def linear(x: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    x: [N, F]
    weight: [F2, F]
    bias: [F2]
    return: [N, F2]
    计算复杂度: O(N F F2)
    """
    res = x @ weight.T
    if bias is not None:
        res.add_(bias)
    return res


# if __name__ == "__main__":
#     x = torch.randn(100, 128)
#     w = torch.randn(256, 128)
#     b = torch.randn(256)
#     ml.test_time(lambda: linear(x, w, b), number=10)
#     ml.test_time(lambda: F.linear(x, w, b), number=10)


def rnn_relu_cell(
    x: Tensor, hx: Tensor,
    w_ih: Tensor, w_hh: Tensor,
    b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None
) -> Tensor:
    """
    x: [N, Cin]
    hx: [N, Ch]
    w_ih: [Ch, Cin]
    w_hh: [Ch, Ch]
    b_ih: [Ch]
    b_hh: [Ch]
    return: [N, Ch]
    """
    return F.linear(x, w_ih, b_ih).add_(F.linear(hx, w_hh, b_hh)).relu_()


def rnn_tanh_cell(
    x: Tensor, hx: Tensor,
    w_ih: Tensor, w_hh: Tensor,
    b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None
) -> Tensor:
    return F.linear(x, w_ih, b_ih).add_(F.linear(hx, w_hh, b_hh)).tanh_()


# if __name__ == "__main__":
#     x = torch.randn(16, 32)
#     hx = torch.randn(16, 64)
#     w_ih = torch.randn(64, 32)
#     w_hh = torch.randn(64, 64)
#     b_ih = torch.randn(64)
#     b_hh = torch.randn(64)
#     #
#     y = ml.test_time(lambda: rnn_relu_cell(x, hx, w_ih, w_hh, b_ih, b_hh))
#     y2 = ml.test_time(lambda:torch.rnn_relu_cell(x, hx, w_ih, w_hh, b_ih, b_hh))
#     print(torch.allclose(y, y2))
#     y = ml.test_time(lambda: rnn_tanh_cell(x, hx, w_ih, w_hh, b_ih, b_hh))
#     y2 = ml.test_time(lambda:torch.rnn_tanh_cell(x, hx, w_ih, w_hh, b_ih, b_hh))
#     print(torch.allclose(y, y2))


def lstm_cell(
        x: Tensor, hx: Tuple[Tensor, Tensor],
        w_ih: Tensor, w_hh: Tensor,
        b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
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
#     y1 = ml.test_time(lambda: torch.lstm_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     y2 = ml.test_time(lambda: lstm_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))
#     #
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256), torch.randn(100, 256)
#     w_ih = torch.randn(4 * 256, 128)
#     w_hh = torch.randn(4 * 256, 256)
#     y1 = ml.test_time(lambda: torch.lstm_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     y2 = ml.test_time(lambda: lstm_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))


def gru_cell(
        x: Tensor, hx: Tensor,
        w_ih: Tensor, w_hh: Tensor,
        b_ih: Optional[Tensor] = None, b_hh: Optional[Tensor] = None
) -> Tensor:
    """
    x: [N, Cin]
    hx: [N, Ch]
    w_ih: [3*Ch, Cin]. r,z,n
    w_hh: [3*Ch, Ch].
    b_ih: [3*Ch].
    b_hh: [3*Ch].
    return: y: [N, Ch]. y也可以理解为hx_
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
#     y1 = ml.test_time(lambda: torch.gru_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     y2 = ml.test_time(lambda: gru_cell(
#         x, xh, w_ih, w_hh, b_ih, b_hh), number=10)
#     print(torch.allclose(y1[0],  y2[0], atol=1e-6))
#     print(torch.allclose(y1[1],  y2[1], atol=1e-6))
#     #
#     x = torch.randn(100, 128)
#     xh = torch.randn(100, 256)
#     w_ih = torch.randn(3 * 256, 128)
#     w_hh = torch.randn(3 * 256, 256)
#     y1 = ml.test_time(lambda: torch.gru_cell(
#         x, xh, w_ih, w_hh, None, None), number=10)
#     y2 = ml.test_time(lambda: gru_cell(
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
    res, W = _scaled_dot_product_attention(Q, K, V, attn_mask, dropout_p, training)
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
#     ml.seed_everything(42)
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

#     ml.seed_everything(42)
#     y1 = ml.test_time(lambda: multi_head_attention_forward(
#         query, key, value, num_heads,
#         in_proj_weight, in_proj_bias, 0.1,
#         out_proj_weight, out_proj_bias, True,
#         key_padding_mask, True, attn_mask), number=10, warmup=1)
#     ml.seed_everything(42)
#     y2 = ml.test_time(lambda: F.multi_head_attention_forward(
#         query, key, value, embed_dim_to_check, num_heads,
#         in_proj_weight, in_proj_bias, None, None, False, 0.1,
#         out_proj_weight, out_proj_bias, True,
#         key_padding_mask, True, attn_mask), number=10, warmup=1)
#     print(torch.allclose(y1[0], y2[0], atol=1e-6))
#     print(torch.allclose(y1[1], y2[1], atol=1e-6))


def adaptive_avg_pool2d(x: Tensor, output_size: Tuple[int, int]) -> Tensor:
    """
    x: [N, C, Hin, Win]
    output_size: Tuple[Hout, Wout]
    return: [N, C, Hout, Wout]
    """
    N, C, Hin, Win = x.shape
    Hout, Wout = output_size
    #
    split_h = torch.linspace(0, Hin, Hout + 1)
    h_start, h_end = split_h[:-1].long(), split_h[1:].ceil().long()
    split_w = torch.linspace(0, Win, Wout + 1)
    w_start, w_end = split_w[:-1].long(), split_w[1:].ceil().long()
    res = []
    for i in range(Hout):
        for j in range(Wout):
            h_pos, w_pos = slice(int(h_start[i]), int(h_end[i])), slice(int(w_start[j]), int(w_end[j]))
            res.append(torch.mean(x[:, :, h_pos, w_pos], dim=(2, 3)))
    res = torch.stack(res, dim=-1).view(N, C, Hout, Wout)
    return res


# if __name__ == "__main__":
#     x = torch.randn(16, 3, 124, 125)
#     y = ml.test_time(lambda: adaptive_avg_pool2d(x, (101, 102)))
#     y2 = ml.test_time(lambda: F.adaptive_avg_pool2d(x, (101, 102)))
#     print(torch.allclose(y, y2, atol=1e-6))


def adaptive_max_pool2d(x: Tensor, output_size: Tuple[int, int]) -> Tensor:
    """
    x: [N, C, Hin, Win]
    output_size: Tuple[Hout, Wout]
    return: [N, C, Hout, Wout]
    """
    N, C, Hin, Win = x.shape
    Hout, Wout = output_size
    #
    split_h = torch.linspace(0, Hin, Hout + 1)
    h_start, h_end = split_h[:-1].long(), split_h[1:].ceil().long()
    split_w = torch.linspace(0, Win, Wout + 1)
    w_start, w_end = split_w[:-1].long(), split_w[1:].ceil().long()
    res = []
    for i in range(Hout):
        for j in range(Wout):
            h_pos, w_pos = slice(int(h_start[i]), int(h_end[i])), slice(int(w_start[j]), int(w_end[j]))
            res.append(torch.max(x[:, :, h_pos, w_pos].flatten(2, 3), dim=2)[0])
    res = torch.stack(res, dim=-1).view(N, C, Hout, Wout)
    return res

# if __name__ == "__main__":
#     x = torch.randn(16, 3, 124, 125)
#     y = ml.test_time(lambda: adaptive_max_pool2d(x, (101, 102)))
#     y2 = ml.test_time(lambda: F.adaptive_max_pool2d(x, (101, 102)))
#     print(torch.allclose(y, y2))
