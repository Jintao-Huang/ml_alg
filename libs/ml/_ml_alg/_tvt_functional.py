# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

"""
only support tvtFt
"""

from numpy import ndarray
from PIL import Image
import torch
from typing import List, Union, Any, Callable, Optional, Tuple, Literal
from torch import Tensor
import torchvision.transforms as tvt
from torchvision.transforms.functional import to_tensor as _to_tensor, InterpolationMode
import torchvision.transforms.functional as tvtF
import torchvision.transforms.functional_tensor as tvtFt

__all__ = []

def compose(x: Any, transforms: List[Callable[[Any], Tensor]]) -> Tensor:
    for transform in transforms:
        x = transform(x)
    return x


def random_horizontal_flip(x: Tensor, p: float = 0.5) -> Tensor:
    """
    x: [..., H, W]. uint8/float32
    p: flip的概率
    """
    if torch.rand(()) < p:
        x = tvtFt.hflip(x)
    return x


# if __name__ == "__main__":
#     import mini_lightning as ml
#     x = torch.rand(16, 3, 100, 100)
#     ml.seed_everything(42)
#     y = tvt.RandomHorizontalFlip(0.1)(x)
#     ml.seed_everything(42)
#     y2 = random_horizontal_flip(x, 0.1)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, x))
#     #
#     ml.seed_everything(42)
#     y = tvt.RandomHorizontalFlip(0.9)(x)
#     ml.seed_everything(42)
#     y2 = random_horizontal_flip(x, 0.9)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, x))


def _get_ijhw(
    size: Tuple[int, int],  # H, W
    scale: Tuple[float, float],
    ratio: Tuple[float, float]  # W/H
) -> Tuple[int, int, int, int]:  # i, j, h, w
    h, w = size
    area = h * w
    log_ratio = torch.tensor(ratio).log_()
    for _ in range(10):
        # t: target
        scale_t: Tensor = torch.empty(()).uniform_(scale[0], scale[1])
        area_t_sqrt: float = scale_t.mul_(area).sqrt_().item()
        ratio_t_sqrt: float = torch.empty(()).uniform_(log_ratio[0], log_ratio[1]).exp_().sqrt_().item()
        # w_t/h_t=ratio_t; w_t*h_t=area_t
        w_t = round(area_t_sqrt * ratio_t_sqrt)
        h_t = round(area_t_sqrt / ratio_t_sqrt)
        if 0 < w_t <= w and 0 < h_t <= h:
            i = torch.randint(0, h - h_t + 1, ()).item()
            j = torch.randint(0, w - w_t + 1, ()).item()
            return i, j, h_t, w_t
    # center crop. (no random)
    img_ratio = w / h
    w_t, h_t = w, h
    if img_ratio < min(ratio):  # h太长
        h_t = round(w_t / min(ratio))
    elif img_ratio > max(ratio):
        w_t = round(h_t * max(ratio))
    #
    i = (h - h_t) // 2
    j = (w - w_t) // 2
    return i, j, h_t, w_t


def random_resized_crop(
    x: Tensor,
    size: Union[int, Tuple[int, int]],
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),  # W/H
    #
    interpolation: Literal[InterpolationMode.NEAREST, InterpolationMode.BILINEAR,
                           InterpolationMode.BICUBIC] = InterpolationMode.BILINEAR,
    antialias: Optional[bool] = None,
) -> Tensor:
    """
    x: [N, C, H, W] or [C, H, W]
    size: int/len=1: [size, size]; len=2: [H, W]. (与resized_crop不同)
    scale: 面积的尺寸的比例随机(使用uniform)
    ratio: 裁剪的W/H比例随机范围(使用log uniforms)
    # 
    antialias: 只适用于BILINEAR, BICUBIC. 使得输出与PIL更接近 
    """
    if isinstance(size, int):
        size = (size, size)
    #
    img_size = x.shape[-2:]
    i, j, h, w = _get_ijhw(img_size, scale, ratio)
    #
    img = tvtFt.crop(x, i, j, h, w)
    img = tvtFt.resize(img, size, interpolation.value, antialias=antialias)
    return img


if __name__ == "__main__":
    import mini_lightning as ml
    x = torch.randn(16, 3, 100, 100)
    ml.seed_everything(42)
    y = tvt.RandomResizedCrop((32, 32))(x)
    ml.seed_everything(42)
    y2 = random_resized_crop(x, (32, 32))
    print(torch.allclose(y, y2))


def random_apply():
    pass


def color_jitter():
    pass


def random_gray_scale():
    pass


def gaussian_blur():
    pass


def to_tensor(x: Union[ndarray, Image.Image]) -> Tensor:
    """
    x: ndarray[H, W, C] or [H, W], uint8/float32. Image.Image["RGB", "L"]
        若为float64, 则输出float64. 所以一般传入uint8. 会进行除以255处理
    return: [C, H, W] or [1, H, W]
    """
    return _to_tensor(x)


# if __name__ == "__main__":
#     import numpy as np
#     x = np.random.rand(100, 100)
#     y = tvt.ToTensor()(x)
#     y2 = to_tensor(x)
#     print(torch.allclose(y, y2))


def normalize(x: Tensor, mean: List[float], std: List[float], inplace: bool = False) -> Tensor:
    """
    x: [..., C, H, W]. C=1 or 3. float32
    mean, std: len=C or 1(广播)
    return: [..., C, H, W]
    """
    return tvtFt.normalize(x, mean, std, inplace)


# if __name__ == "__main__":
#     x = torch.rand(3, 100, 100)
#     y = tvt.Normalize((0.5,), (0.5,))(x)
#     y2 = normalize(x, (0.5,), (0.5,))
#     print(torch.allclose(y, y2))
#     #
#     x = torch.rand(16, 3, 100, 100)
#     y = tvt.Normalize((0.5,), (0.5,))(x)
#     y2 = normalize(x, (0.5,), (0.5,))
#     print(torch.allclose(y, y2))
