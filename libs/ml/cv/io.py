import cv2 as cv
import numpy as np
from numpy import ndarray
from PIL import Image
import torch
from torch import Tensor

__all__ = ["imwrite", "imread", "pil_to_cv", "cv_to_pil",
           "ndarray_to_tensor", "tensor_to_ndarray"]


def imwrite(arr: ndarray, fpath: str) -> bool:
    """cv无法读取中文字符. 此文件用于写入中文字符路径的文件"""
    # 使用np的保存文件的方式
    ext = fpath.rsplit('.', 1)[1]
    retval, img = cv.imencode(f".{ext}", arr)  # img: ndim=1
    if retval is True:
        img.tofile(fpath)
    return retval


def imread(fpath: str, flags=cv.IMREAD_COLOR) -> ndarray:
    """cv无法读取中文字符. 此文件用于读取中文字符路径的文件"""
    # 使用numpy的读取文件的方法
    # cv.IMREAD_UNCHANGED: -1
    # cv.IMREAD_GRAYSCALE: 0
    # cv.IMREAD_COLOR: 1
    img = np.fromfile(fpath, dtype=np.uint8)  # img: ndim=1
    return cv.imdecode(img, flags)


# if __name__ == "__main__":
#     img_fname = "asset/哈哈.png"
#     img = imread(img_fname)
#     cv.imshow("1", img)
#     cv.waitKey(0)
#     imwrite(img, "asset/1.png")


def pil_to_cv(img_pil: Image.Image, to_bgr=True) -> ndarray:
    # pil_RGB to cv_BGR
    mode = img_pil.mode
    img = np.asarray(img_pil)  # type: ndarray
    if mode == "RGB":
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR) if to_bgr else img
    elif mode == "RGBA":
        img = cv.cvtColor(img, cv.COLOR_RGBA2BGRA) if to_bgr else img
    elif mode == "L":
        img = img  # [H, W]
    else:
        raise ValueError(f"mode: {mode}")
    return img


def cv_to_pil(img: ndarray, is_bgr=True) -> Image.Image:
    # cv_BGR to pil_RGB
    if img.ndim == 2:
        mode = "L"
    elif img.ndim == 3 and img.shape[2]:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if is_bgr else img
        mode = "RGB"
    elif img.ndim == 3 and img.shape[2]:
        img = cv.cvtColor(img, cv.COLOR_BGRA2RGBA) if is_bgr else img
        mode = "RGBA"
    else:
        raise ValueError(f"img.shape: {img.shape}")
    return Image.fromarray(img, mode)

# if __name__ == "__main__":
#     img_fname = "asset/哈哈.png"
#     img = Image.open(img_fname)
#     print(img.size)
#     img=img.convert("L")
#     img2= pil_to_cv(img)
#     print(img2.shape)
#     img = cv_to_pil(img2)
#     img.show()


def _ndarray_to_tensor(arr: ndarray, to_float: bool = True, is_bgr=False) -> Tensor:
    # arr: [N, H, W, C] -> tensor: [N, C, H, W]
    tensor = torch.from_numpy(arr)
    tensor = tensor.permute(0, 3, 1, 2)
    if tensor.dtype != torch.float32 and to_float:
        tensor = tensor.to(dtype=torch.float32)
        tensor /= 255
    tensor = tensor.flip(dims=(1,)) if is_bgr else tensor
    return tensor


def _tensor_to_ndarray(tensor: Tensor, to_uint8: bool = True, to_bgr=False) -> ndarray:
    # tensor: [N, C, H, W] -> arr: [N, H, W, C]
    tensor = tensor.flip(dims=(1,)) if to_bgr else tensor
    if tensor.dtype != torch.uint8 and to_uint8:
        tensor *= 255
        tensor = tensor.to(torch.uint8)
    tensor = tensor.permute(0, 2, 3, 1)
    arr = tensor.numpy()  # type: ndarray
    return arr


def ndarray_to_tensor(arr: ndarray, to_float: bool = True, is_bgr=False):
    # RGB不变
    if arr.ndim == 3:
        arr = arr[None]
        tensor = _ndarray_to_tensor(arr, to_float, is_bgr)[0]
    elif arr.ndim == 4:
        tensor = _ndarray_to_tensor(arr, to_float, is_bgr)
    else:
        raise ValueError(f"arr.ndim: {arr.ndim}")
    return tensor


def tensor_to_ndarray(tensor: Tensor, to_uint8: bool = True, to_bgr=False):
    # RGB不变
    if tensor.ndim == 3:
        tensor = tensor[None]
        arr = _tensor_to_ndarray(tensor, to_uint8, to_bgr)[0]
    elif tensor.ndim == 4:
        arr = _tensor_to_ndarray(tensor, to_uint8, to_bgr)
    else:
        raise ValueError(f"tensor.ndim: {tensor.ndim}")
    return arr


# if __name__ == "__main__":
#     x = np.random.randint(0, 256, (200, 200, 3))
#     print(x.dtype, x.shape, x.mean(axis=(0, 1)))
#     x_t = ndarray_to_tensor(x, is_bgr=True)
#     x = x_t.numpy()
#     print(x.dtype, x.shape, x.mean(axis=(1, 2)) * 255)
#     x = tensor_to_ndarray(x_t, to_bgr=True)
#     print(x.dtype, x.shape, x.mean(axis=(0, 1)))
