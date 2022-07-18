

import torchvision.transforms.functional_tensor as tvF_t
import torchvision.transforms.functional_pil as tvF_pil
import torchvision.transforms.functional as tvF
import torchvision.transforms as tvt
from torchvision.transforms.functional import InterpolationMode, pil_modes_mapping
from torch import Tensor
from torchvision.datasets import CIFAR10
from typing import List, Tuple
import torch
import os


def random_horizontal_flip(p: float):
    def _(x):
        if torch.rand(1) < p:
            x = tvF_pil.hflip(x)
        return x
    return _


def random_resized_crop(size: Tuple[int, int], scale: Tuple[float, float],
                        ratio: Tuple[float, float], interpolation=InterpolationMode.BILINEAR):
    # size: crop后进行resize, 和随机性无关
    # scale代表面积的范围; ratio代表长宽比的范围
    def _(x):
        # 传入x, 为了获取x的shape.
        l, t, h, w = tvt.RandomResizedCrop.get_params(x, scale, ratio)
        x = tvF_pil.crop(x, l, t, h, w)
        x = tvF_pil.resize(
            x, size, interpolation=pil_modes_mapping[interpolation])
        return x
    return _


# if __name__ == "__main__":
#     DATASETS_PATH = os.environ.get("DATASETS_PATH")
#     train_dataset = CIFAR10(root=DATASETS_PATH, train=True, download=True)
#     DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0, 1, 2))
#     DATA_STD = (train_dataset.data / 255.0).std(axis=(0, 1, 2))
#     torch.manual_seed(42)
#     train_transform = tvt.Compose(
#         [
#             tvt.RandomHorizontalFlip(),
#             tvt.RandomResizedCrop((32, 32), scale=(
#                 0.8, 1.0), ratio=(0.9, 1.1)),
#             tvt.ToTensor(),
#             tvt.Normalize(DATA_MEANS, DATA_STD),
#         ]
#     )
#     y1 = train_transform(train_dataset[0][0])
#     torch.manual_seed(42)
#     train_transform2 = tvt.Compose(
#         [
#             random_horizontal_flip(0.5),
#             random_resized_crop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
#             tvF.to_tensor,
#             lambda x: tvF.normalize(x, DATA_MEANS, DATA_STD),
#         ]
#     )
#     y2 = train_transform2(train_dataset[0][0])
#     print(torch.allclose(y1, y2))  # True
