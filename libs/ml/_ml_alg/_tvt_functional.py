# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from torchvision.transforms.functional_tensor import _rgb2hsv, _hsv2rgb
from typing import Union, List, Literal, Tuple, Optional
from torch import Tensor
import torch
from numpy import ndarray
from PIL import Image
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as tvtF
__all__ = []


if __name__ == "__main__":
    import mini_lightning as ml


def to_tensor(x: Union[ndarray, Image.Image]) -> Tensor:
    """
    x: ndarray[H, W, C] uint8/Any. Image.Image["RGB"]
    return: Tensor[C, H, W]. if x is uint8, 则输出除以255 -> [0..1]
    """
    if isinstance(x, Image.Image):
        if x.mode != "RGB":  # 这里只支持RGB
            raise ValueError(f"x.mode: {x.mode}")
        x = np.array(x, dtype=np.uint8)
    #
    if not isinstance(x, ndarray):
        raise ValueError(f"type(x): {type(x)}")
    #
    t = torch.from_numpy(x)
    if t.dtype == torch.uint8:
        t = t.to(torch.float32).div_(255)
    t = t.permute(2, 0, 1)
    return t


# if __name__ == "__main__":
#     x = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
#     y = ml.test_time(lambda: tvtF.to_tensor(x))
#     y2 = ml.test_time(lambda: to_tensor(x))
#     print(torch.allclose(y, y2))
#     #
#     x = Image.fromarray(x)
#     y = ml.test_time(lambda: tvtF.to_tensor(x))
#     y2 = ml.test_time(lambda: to_tensor(x))
#     print(torch.allclose(y, y2))


def normalize(x: Tensor, mean: Union[List[float], Tensor], std: Union[List[float], Tensor],
              inplace: bool = False) -> Tensor:
    """
    x: [..., C, H, W]. float32/64
    mean, std: len=C or 1(广播)
    """
    dtype, device = x.dtype, x.device
    assert dtype in {torch.float32, torch.float64}
    mean = torch.as_tensor(mean, dtype=dtype, device=device)
    std = torch.as_tensor(std, dtype=dtype, device=device)
    if not inplace:
        x = x.clone()
    mean = mean[:, None, None]
    std = std[:, None, None]
    return x.sub_(mean).div_(std)


# if __name__ == "__main__":
#     x = torch.rand(3, 100, 100)
#     y = ml.test_time(lambda: tvtF.normalize(x, [0.2, 0.4, 0.1], [0.3, 0.1, 0.2]))
#     y2 = ml.test_time(lambda: normalize(x, [0.2, 0.4, 0.1], [0.3, 0.1, 0.2]))
#     print(torch.allclose(y, y2, atol=1e-6))
#     #
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.normalize(x, [0.2], [0.3]))
#     y2 = ml.test_time(lambda: normalize(x, [0.2], [0.3]))
#     print(torch.allclose(y, y2, atol=1e-6))


def pad(x: Tensor, padding: Union[int, List[int]], fill: float = 0.,
        padding_mode: Literal["constant"] = "constant") -> Tensor:
    """
    x: [..., H, W]. dtype: Any
    padding: 
        int/Tuple[int]: ltrb
        Tuple[int, int]: lr,tb
        Tuple[int, int, int, int]: l,t,r,b
    """
    #
    if isinstance(padding, int):
        l = r = t = b = padding
    else:
        if len(padding) == 1:
            l = r = t = b = padding[0]
        elif len(padding) == 2:
            l = r = padding[0]
            t = b = padding[1]
        elif len(padding) == 4:
            l, t, r, b = padding
        else:
            raise ValueError(f"len(padding): {len(padding)}")
    padding = [l, r, t, b]
    return F.pad(x, padding, padding_mode, fill)


# if __name__ == "__main__":
#     x = torch.rand(100, 100)
#     y = ml.test_time(lambda: tvtF.pad(x, [2], 1))
#     y2 = ml.test_time(lambda: pad(x, [2], 1))
#     print(torch.allclose(y, y2))
#     #
#     x = torch.rand(3, 100, 100).mul(255).to(torch.uint8)
#     y = ml.test_time(lambda: tvtF.pad(x, [2, 3], 1))
#     y2 = ml.test_time(lambda: pad(x, [2, 3], 1))
#     print(torch.allclose(y, y2))
#     #
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.pad(x, [2, 3, 4, 5], 1))
#     y2 = ml.test_time(lambda: pad(x, [2, 3, 4, 5], 1))
#     print(torch.allclose(y, y2))


def hflip(x: Tensor) -> Tensor:
    """
    x: [..., H, W]. Any
    """
    return x.flip(-1)


def vflip(x: Tensor) -> Tensor:
    """
    x: [..., H, W]. Any
    """
    return x.flip(-2)


# if __name__ == "__main__":
#     x = torch.rand(100, 100)
#     y = ml.test_time(lambda: tvtF.hflip(x))
#     y2 = ml.test_time(lambda: hflip(x))
#     print(torch.allclose(y, y2))
#     #
#     y = ml.test_time(lambda: tvtF.vflip(x))
#     y2 = ml.test_time(lambda: vflip(x))
#     print(torch.allclose(y, y2))


def rgb_to_grayscale(x: Tensor, num_output_channels: int = 1) -> Tensor:
    """
    x: [..., 3, H, W]. float*/uint8/Any
    return: [..., num_output_channels, H, W]
    """
    r, g, b = torch.unbind(x, dim=-3)
    dtype = x.dtype
    res = r.mul(0.2989).add_(g.mul(0.587)).add_(b.mul(0.114)).to(dtype).unsqueeze(-3)
    assert num_output_channels in {1, 3}
    if num_output_channels == 3:
        res = res.expand(x.shape)
    return res


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda:tvtF.rgb_to_grayscale(x, 1))
#     y2 = ml.test_time(lambda:rgb_to_grayscale(x, 1))
#     print(torch.allclose(y, y2))
#     x = torch.rand(16, 3, 100, 100).mul(255).to(torch.uint8)
#     y = ml.test_time(lambda:tvtF.rgb_to_grayscale(x, 3))
#     y2 = ml.test_time(lambda:rgb_to_grayscale(x, 3))


def crop(x: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """
    x: [..., H, W]. 若超出边界, 则pad0.
    return: [..., max(height, 0), max(width, 0)]
    """
    h, w = x.shape[-2:]
    t, l, b, r = top, left, top+height, left+width
    del top, left, height, width
    x = x[..., max(t, 0):max(0, b), max(l, 0):max(0, r)]  # torch支持负数索引. 所以使其>=0
    if t >= 0 and l >= 0 and b < h and r < w:
        return x
    #
    padding = [  # l,t,r,b
        max(-l + min(0, r), 0),   # min: 避免l,r<0; max: 避免r<l<0
        max(-t + min(0, b), 0),
        max(r - max(w, l), 0),  # 内max: 避免l,r>w; 外max: 避免w<r<l
        max(b - max(h, t), 0)
    ]
    return pad(x, padding, 0.)


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.crop(x, 10, 10, 200, 50))
#     y2 = ml.test_time(lambda: crop(x, 10, 10, 200, 50))
#     print(torch.allclose(y, y2), y.shape)
#     #
#     y = ml.test_time(lambda: tvtF.crop(x, -20, -20, 10, 10))
#     y2 = ml.test_time(lambda: crop(x, -20, -20, 10, 10))
#     print(y.shape, y2.shape)
#     #
#     y = ml.test_time(lambda: tvtF.crop(x, 200, 200, 100, 100))
#     y2 = ml.test_time(lambda: crop(x, 200, 200, 100, 100))
#     print(torch.allclose(y, y2), y.shape)
#     #
#     y = ml.test_time(lambda: tvtF.crop(x, 200, 200, -50, -50))
#     y2 = ml.test_time(lambda: crop(x, 200, 200, -50, -50))
#     print(torch.allclose(y, y2), y.shape)
#     #
#     y = ml.test_time(lambda: tvtF.crop(x, -20, -20, -10, -10))
#     y2 = ml.test_time(lambda: crop(x, -20, -20, -10, -10))
#     print(y.shape, y2.shape)

def center_crop(x: Tensor, output_size: Union[int, List[int]]) -> Tensor:
    """
    x: [..., H, W]. 若超出, 则pad0
    output_size: int/len=1: H=W; len=2: [H, W]
    return 
    """
    h_img, w_img = x.shape[-2:]
    if isinstance(output_size, int):
        h_out = w_out = output_size
    elif len(output_size) == 1:
        h_out = w_out = output_size[0]
    elif len(output_size) == 2:
        h_out, w_out = output_size
    else:
        raise ValueError(f"len(output_size): {len(output_size)}")
    #
    t, l = round((h_img - h_out) / 2), round((w_img - w_out) / 2)
    return crop(x, t, l, h_out, w_out)


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.center_crop(x, [200, 50]))
#     y2 = ml.test_time(lambda: center_crop(x, [200, 50]))
#     print(torch.allclose(y, y2))
#     #
#     y = ml.test_time(lambda: tvtF.center_crop(x, [50, 200]))
#     y2 = ml.test_time(lambda: center_crop(x, [50, 200]))
#     print(torch.allclose(y, y2))


def _compute_resized_output_size(
    img_size: Tuple[int, int],
    output_size: Union[int, List[int]],
    max_size: Optional[int] = None
) -> Tuple[int, int]:
    """
    img_size: [H, W]
    output_size: int/len=1: 最短边; len=2: [H, W]
    max_size: 只有当output_size为int/len=1时有效
    # 
    note: 这里使用round, 而不是floor. 我认为更加科学, 不同于tvtF._compute_resized_output_size
    """
    if isinstance(output_size, int):
        output_size = [output_size]

    if len(output_size) == 1:
        # max_size优先级比output_size高. 所以先计算output_size, 再计算max_size
        s_limit = output_size[0]  # short 限制
        h, w = img_size
        l, s = (h, w) if h >= w else (w, h)  # long short
        d = s_limit / s
        #
        if max_size is not None:  # l_limit
            dl = max_size / l
            d = min(d, dl)
        #
        new_h, new_w = round(d * h), round(d * w)
    else:
        assert max_size is None
        new_h, new_w = output_size
    return new_h, new_w


# if __name__ == "__main__":
#     img_size = (100, 124)
#     output_size = [145]
#     max_size = 146
#     print(tvtF._compute_resized_output_size(img_size, output_size, max_size))
#     print(_compute_resized_output_size(img_size, output_size, max_size))


def resize(
    x: Tensor,
    size: Union[int, List[int]],
    interpolation: Literal[tvtF.InterpolationMode.NEAREST,
                           tvtF.InterpolationMode.BILINEAR,
                           tvtF.InterpolationMode.BICUBIC] = tvtF.InterpolationMode.BILINEAR,
    max_size: Optional[int] = None,
    antialias: Optional[bool] = None
) -> Tensor:
    """
    x: [..., H, W]. float32/64
    size: int/len=1: 最短边(保持aspect ratio); len=2: [H, W]
    interpolation: Tensor下只支持这三种
    """
    assert x.dtype in {torch.float32, torch.float64}
    #
    H, W = x.shape[-2:]
    output_size = _compute_resized_output_size((H, W), size, max_size)
    #
    if antialias is None:
        antialias = False
    align_corners = None
    if interpolation in {tvtF.InterpolationMode.BILINEAR, tvtF.InterpolationMode.BICUBIC}:
        align_corners = False
    #
    need_squeeze = False
    if interpolation == tvtF.InterpolationMode.BICUBIC and x.ndim == 3:
        x = x[None]
        need_squeeze = True
    x = F.interpolate(x, size=output_size, mode=interpolation.value, align_corners=align_corners, antialias=antialias)
    if need_squeeze:
        x = x[0]
    return x


# if __name__ == "__main__":
#     x = torch.rand(3, 124, 200)
#     y = ml.test_time(lambda: tvtF.resize(x, 100, tvtF.InterpolationMode.BICUBIC, 120))
#     y2 = ml.test_time(lambda: resize(x, 100, tvtF.InterpolationMode.BICUBIC, 120))
#     print(torch.allclose(y, y2), y.shape)


def resized_crop(
    x: Tensor,
    top: int, left: int, height: int, width: int,
    size: Union[int, List[int]],
    interpolation: Literal[tvtF.InterpolationMode.NEAREST,
                           tvtF.InterpolationMode.BILINEAR,
                           tvtF.InterpolationMode.BICUBIC] = tvtF.InterpolationMode.BILINEAR,
    antialias: Optional[bool] = None
) -> Tensor:
    """
    x: [..., H, W]
    """
    x = crop(x, top, left, height, width)
    x = resize(x, size, interpolation, antialias=antialias)
    return x


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 200)
#     y = ml.test_time(lambda: tvtF.resized_crop(x, 10, 10, 200, 50, 100))
#     y2 = ml.test_time(lambda: resized_crop(x, 10, 10, 200, 50, 100))
#     print(torch.allclose(y, y2), y.shape)
#     #
#     y = ml.test_time(lambda: tvtF.resized_crop(x, -20, -20, 50, 50, [100, 100]))
#     y2 = ml.test_time(lambda: resized_crop(x, -20, -20, 50, 50, [100, 100]))
#     print(torch.allclose(y, y2), y.shape)
#     #
#     y = ml.test_time(lambda: tvtF.resized_crop(x, 200, 200, 50, 100, [100]))
#     y2 = ml.test_time(lambda: resized_crop(x, 200, 200, 50, 100, [100]))
#     print(torch.allclose(y, y2), y.shape)


def _blend(img: Tensor, img2: Tensor, ratio: float) -> Tensor:
    """img使用ratio, img2使用1-ratio 然后进行混合
    img: [...]. float
    img2: [...]
    return: [...]. 0-bound
    """
    assert img.dtype in {torch.float32, torch.float64}
    assert img.dtype == img2.dtype
    bound = 1
    #
    return img.mul(ratio).add_(img2.mul(1 - ratio)).clamp_(0, bound)


def adjust_brightness(x: Tensor, brightness_factor: float) -> Tensor:
    """
    x: [..., 1 or 3, H, W]. float
    """
    C = x.shape[-3]
    assert C in {1, 3}
    assert brightness_factor >= 0
    img2 = torch.zeros_like(x)
    return _blend(x, img2, brightness_factor)


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.adjust_brightness(x, 2))
#     y2 = ml.test_time(lambda: adjust_brightness(x, 2))
#     print(torch.allclose(y, y2))


def adjust_contrast(x: Tensor,  contrast_factor: float) -> Tensor:
    """
    x: [..., 1 or 3, H, W]. float
    """
    C = x.shape[-3]
    assert C in {1, 3}
    assert contrast_factor >= 0
    #
    gray = x
    if C == 3:
        gray = rgb_to_grayscale(x, 1)
    gray_mean = gray.mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(x, gray_mean, contrast_factor)


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.adjust_contrast(x, 2))
#     y2 = ml.test_time(lambda: adjust_contrast(x, 2))
#     print(torch.allclose(y, y2))
#     x = torch.randint(0, 256, (16, 3, 100, 100), dtype=torch.uint8)
#     y = ml.test_time(lambda: tvtF.adjust_contrast(x, 2))
#     # 我们的adjust_contrast只支持了float


def adjust_saturation(x: Tensor, contrast_factor: float) -> Tensor:
    """
    x: [..., 1 or 3, H, W]. float
    """
    C = x.shape[-3]
    assert C in {1, 3}
    assert contrast_factor >= 0
    #
    gray = x
    if C == 1:
        return x
    #
    gray = rgb_to_grayscale(x, 1)
    return _blend(x, gray, contrast_factor)


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.adjust_saturation(x, 2))
#     y2 = ml.test_time(lambda: adjust_saturation(x, 2))
#     print(torch.allclose(y, y2))

# def _rgb2hsv(x: Tensor) -> Tensor:
#     pass


# def _hsv_2rgb(x: Tensor) -> Tensor:
#     pass


def adjust_hue(x: Tensor, hue_factor: float) -> Tensor:
    """
    x: [..., 1 or 3, H, W]. float. >=0
    """
    C = x.shape[-3]
    assert C in {1, 3}
    assert -0.5 <= hue_factor <= 0.5
    #
    if C == 1:
        return x
    #
    x = _rgb2hsv(x)
    x[:, 0] = x[:, 0].add_(hue_factor).remainder_(1)
    res = _hsv2rgb(x)
    return res


# if __name__ == "__main__":
#     x = torch.rand(16, 3, 100, 100)
#     y = ml.test_time(lambda: tvtF.adjust_hue(x, -0.3), 5)
#     y2 = ml.test_time(lambda: adjust_hue(x, -0.3), 5)
#     print(torch.allclose(y, y2))


def rotate():
    pass


def affine():
    pass
