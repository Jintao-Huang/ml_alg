# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ...._types import *
# from libs import *


def _nearest_interpolate(x: Tensor, size: Tuple[int, int]) -> Tensor:
    """
    x: [N, C, Hin, Win]
    size: Tuple[Hout, Wout]
    return: [N, C, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    Hout, Wout = size
    #
    grid_h = torch.linspace(0, Hin, Hout + 1)[:-1].long()  # [Hout]
    grid_w = torch.linspace(0, Win, Wout + 1)[:-1].long()  # [Wout]
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")  # [Hout, Wout]
    return x[:, :, grid_h, grid_w]


# if __name__ == "__main__":
#     x = torch.randn(16, 3, 124, 125)
#     y = ml.test_time(lambda: _nearest_interpolate(x, (101, 102)))
#     y2 = ml.test_time(lambda: F.interpolate(x, (101, 102)))
#     print(torch.allclose(y, y2), y.shape)


def _bilinear_interpolate(x: Tensor, size: Tuple[int, int], align_corners: bool = False) -> Tensor:
    """
    x: [N, C, Hin, Win]
    size: Tuple[Hout, Wout]
    return: [N, C, Hout, Wout]
    """
    Hin, Win = x.shape[2:]
    Hout, Wout = size
    #
    if not align_corners:
        grid_h = torch.linspace(0, Hin, Hout + 1).sub_(0.5)
        grid_w = torch.linspace(0, Win, Wout + 1).sub_(0.5)
        step_h, step_w = grid_h[1] - grid_h[0], grid_w[1] - grid_w[0]
        grid_h = grid_h.add_(step_h / 2)[:-1]
        grid_w = grid_w.add_(step_w / 2)[:-1]
        #
        grid_h = grid_h.clamp_(0, Hin - 1)
        grid_w = grid_w.clamp_(0, Win - 1)
    else:
        grid_h = torch.linspace(0, Hin - 1, Hout)
        grid_w = torch.linspace(0, Win - 1, Wout)
    #
    grid_h, grid_w = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid_top, grid_left = grid_h.long(), grid_w.long()
    grid_bottom, grid_right = grid_h.ceil().long(), grid_w.ceil().long()
    #
    offset_h0, offset_w0 = grid_h - grid_top, grid_w - grid_left
    offset_h1, offset_w1 = 1 - offset_h0, 1 - offset_w0

    res = offset_h1.mul(offset_w1).mul(x[:, :, grid_top, grid_left]) + \
        offset_h1.mul(offset_w0).mul(x[:, :, grid_top, grid_right]) + \
        offset_h0.mul(offset_w1).mul(x[:, :, grid_bottom, grid_left]) + \
        offset_h0.mul(offset_w0).mul(x[:, :, grid_bottom, grid_right])
    return res


# if __name__ == "__main__":
#     x = torch.randn(16, 3, 124, 125)
#     y = ml.test_time(lambda: _bilinear_interpolate(x, (101, 102), align_corners=True))
#     y2 = ml.test_time(lambda: F.interpolate(x, (101, 102), mode="bilinear", align_corners=True))
#     print(torch.allclose(y, y2, atol=1e-4))
#     #
#     x = torch.randn(16, 3, 124, 125)
#     y = ml.test_time(lambda: _bilinear_interpolate(x, (101, 102), align_corners=False))
#     y2 = ml.test_time(lambda: F.interpolate(x, (101, 102), mode="bilinear", align_corners=False))
#     print(torch.allclose(y, y2, atol=1e-4))


def interpolate(
    x: Tensor,
    size: Tuple[int, int],
    mode: Literal["nearest", "bilinear", "area"] = "nearest",
    align_corners: Optional[bool] = None
) -> Tensor:
    """
    x: [N, C, Hin, Win]
    size: Tuple[Hout, Wout]
    return: [N, C, Hout, Wout]
    """
    if mode == "nearest":
        return _nearest_interpolate(x, size)
    elif mode == "bilinear":
        align_corners = False if align_corners is None else align_corners
        return _bilinear_interpolate(x, size, align_corners)
    elif mode == "area":
        return F.adaptive_avg_pool2d(x, size)
    else:
        raise ValueError(f"mode: {mode}")


# if __name__ == "__main__":
#     x = torch.randn(16, 3, 124, 125)
#     # y = ml.test_time(lambda: _nearest_interpolate(x, (101, 102), ))
#     y = ml.test_time(lambda: interpolate(x, (101, 102), mode="area"))
#     y2 = ml.test_time(lambda: F.interpolate(x, (101, 102), mode="area"))
#     print(torch.allclose(y, y2))
