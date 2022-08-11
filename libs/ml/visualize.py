# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from torch import device as Device, Tensor
from torch.nn import Module
import torch
from matplotlib.colors import to_rgb
from typing import Optional, List, Union, Callable, Dict, Tuple
from torch.utils.data import DataLoader, TensorDataset
from numpy import ndarray
import math
from torchvision.utils import make_grid as _make_grid
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

__all__ = ["bincount", "plot_classification_map", "visualize_samples",
           "plot_lines", "plot_subplots", "make_grid", "normalize", "make_grid2",
           "tensorboard_smoothing", "config_ax", "read_tensorboard_file"]


def bincount(x: ndarray, n_bin: int = None, step: int = None,
                  min_: int = None, max_: int = None,
                  show: bool = False) -> ndarray:
    """for debug. 
    n_bin, step必须提供一个参数. n_bin表示桶的个数, step表示bin的宽度. 
    """
    min_ = min_ if min_ is not None else x.min()
    max_ = max_ if max_ is not None else x.max()
    if n_bin is not None:
        assert step is None
        step = math.ceil((max_ - min_) / n_bin)
    assert step is not None
    x = np.divide(x, step).astype(np.int32)
    bc = np.bincount(x)
    if show:
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(bc.shape[0]) * step, bc)
        plt.show()
    return bc


@ torch.no_grad()
def plot_classification_map(model: Module, device: Device,
                            extent: Tuple[float, float, float, float], n_labels: int, ax: Axes) -> None:
    """只支持输入为2D的情形"""
    # 功能: 对输入为2D的模型, 遍历输入. 产生输出的map. 使用分类颜色的混合来可视化
    # extent: lrtb
    if n_labels > 10:
        raise ValueError(f"n_labels: {n_labels}")
    ci = torch.empty((n_labels, 3))
    for i in range(n_labels):
        ci[i] = torch.tensor(to_rgb(f"C{i}"))
    x1 = torch.linspace(extent[0], extent[1], 100)
    x2 = torch.linspace(extent[2], extent[3], 100)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='xy')
    x = torch.stack([xx1, xx2], dim=-1)
    shape = x.shape
    x = x.view(-1, 2)
    #
    dataset = TensorDataset(x)
    batch_size = 128
    dataloader = DataLoader(dataset, batch_size, False, drop_last=False)
    y = torch.empty((*x.shape[:-1], n_labels))
    del x, x1, x2, xx1, xx2
    #
    device_r = next(model.parameters()).device
    model.to(device)
    for i, (x_batch,) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = torch.sigmoid(model(x_batch))
        if y_batch.shape[-1] == 1 and n_labels == 2:
            # 二分类的特殊处理
            y_batch = torch.concat([1-y_batch, y_batch], dim=-1)
        _range = slice(i * batch_size, (i+1)*batch_size)
        y[_range] = y_batch.cpu()
    # y, model都在cpu上
    model.to(device_r)  # save memory
    y = y.view(*shape[:2], n_labels)
    # y: [NX, NY, NL]. ci: [NL, 3]
    # res: [NX, NY, 3]
    output_image = y @ ci
    output_image = output_image.numpy()

    ax.imshow(output_image, origin="lower", extent=extent)
    ax.grid(False)


def visualize_samples(data: Union[Tensor, ndarray], labels: Union[Tensor, ndarray], ax: Axes) -> None:
    """data可以先降维后输入. 可支持多分类"""
    # 功能: 将data, labels对应的样本点进行可视化. scatter+legend
    # data是2D的[n,f]的Tensor, f=2. float
    # labels: [n]. int
    if isinstance(data, Tensor):
        data = data.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()
    #
    n_labels = np.max(labels) + 1
    if n_labels > 10:
        raise ValueError(f"n_labels: {n_labels}")
    data_i = [None] * n_labels
    for i in range(n_labels):
        data_i[i] = data[labels == i]
    del data
    #
    for i in range(n_labels):
        ax.scatter(data_i[i][:, 0], data_i[i][:, 1],
                   edgecolor="#333", label=f"Class {i}")
    ax.set_title("Dataset samples")
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    ax.legend()


# if __name__ == "__main__":
#     try:
#         from . import XORDataset, MLP_L2
#     except ImportError:
#         from _trash import MLP_L2, XORDataset

#     dataset = XORDataset(200)
#     fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
#     dataset.labels[:100][dataset.labels[:100] == 1] = 2
#     n_labels = 3
#     model = MLP_L2(2, 4, n_labels)
#     plot_classification_map(model, Device(
#         'cuda'), (-0.5, 1.5, 0, 2), n_labels, ax)
#     visualize_samples(dataset.data, dataset.labels, ax)
#     # plt.savefig("runs/images/1.png", bbox_inches='tight')
#     plt.show()


def plot_lines(funcs: List[Callable[[ndarray], ndarray]], labels: List[str],
               ax: Axes, x_range: Tuple[int, int]) -> None:
    """labels: for legend"""
    # 输入前需要将ax的config设置好: e.g. xlabel, ylabel, xlim, ylim, xticks, yticks, title
    # 功能: 画对应funcs的多条线条(每个对应的label为labels[i]). x_range为对应funcs的x的范围
    x = np.linspace(x_range[0], x_range[1])
    for i, func in enumerate(funcs):
        y = func(x)
        ax.plot(x, y, linewidth=2, color=f"C{i}", label=labels[i])
    ax.legend()


def plot_subplots(plot_funcs: List[Callable[[Axes], None]],
                  ncols: int = 2, fig: Figure = None) -> Figure:
    # 功能: 每个subplots对应plot_funcs. 对每个subplots进行画图
    n = len(plot_funcs)
    nrows = int(math.ceil(n / ncols))
    if fig is None:
        fig = plt.figure(figsize=(4 * ncols, 4 * nrows), dpi=200)
    axs = fig.subplots(nrows, ncols)
    for i in range(nrows):
        axs_r = (axs[i] if nrows > 1 else axs)
        for j in range(ncols):
            idx = i * ncols + j
            if idx >= n:
                continue
            ax = (axs_r[j] if ncols > 1 else axs_r)
            plot_funcs[idx](ax)
    fig.subplots_adjust(hspace=0.3)
    return fig


# if __name__ == "__main__":
#     from functools import partial
#     funcs_s = [[np.sin, np.cos], [lambda x: 2 * x, lambda x: x + 1]]
#     labels_s = [["sin", "cos"], ["2x", "x+1"]]
#     plot_funcs = [partial(plot_lines, funcs, labels, x_range=(-5, 5))
#                   for funcs, labels in zip(funcs_s, labels_s)]
#     # 会出错: 因为闭包的性质. 请使用上面的用法
#     # 当调用函数的时候, funcs, labels都等于funcs_s[-1], labels_s[-1]
#     # plot_funcs = [lambda ax: plot_lines(funcs, labels, ax, x_range=(-5, 5))
#     #               for funcs, labels in zip(funcs_s, labels_s)]

#     plot_subplots(plot_funcs)
#     plt.show()


def make_grid(images: Union[Tensor, ndarray], ax: Axes,
              ncols: int = 4, *, norm: bool = True, pad_value=0.) -> None:
    # Tensor: [N, C, H, W]. 0-1
    # ndarray: [N, H, W, C]. 0-1
    # 将ax传入前, 先进行config_ax
    # 功能: 将图片按网格进行组合.
    if images.ndim != 4:
        raise ValueError(f"images.ndim: {images.ndim}")
    if isinstance(images, ndarray):
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
    # 归一化
    # ncols: 列数
    # normalize: 先将images归一化到0-1之间, 然后pad
    images = _make_grid(images, nrow=ncols, normalize=norm,
                        pad_value=pad_value)  # 返回 [C, H, W], 0-1
    #
    images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()  # share memory
    #
    # 若未指定vim, vmax, 则vmin=images.min(), vmax=images.max()
    # 随后将[vmin, vmax] norm-> [0, 1]
    # images是C=3的, 所以cmap不发挥作用
    ax.imshow(images, cmap=None, origin="upper", vmin=0, vmax=1)
    ax.axis("off")


def normalize(image: Tensor) -> Tensor:
    """线性norm"""
    # image: [N, C, H, W]. 按min, max
    # 这与tvF.normalize不同. 对每个样本norm. tvF.normalize: 对每个通道norm
    # 功能: 归一化到[0..1]. 而不是标准化

    n_samples = image.shape[0]
    #
    ndim = image.ndim
    image_v = image.view(n_samples, -1)
    shape = n_samples,  *(1,)*(ndim-1)  # keepdim
    max_ = image_v.max(dim=1)[0].view(shape)
    min_ = image_v.min(dim=1)[0].view(shape)
    # img_norm * (max_ - min_) + min_ = img
    image -= min_
    image /= (max_ - min_)
    return image


def make_grid2(images: Union[Tensor, ndarray], fig: Figure = None,
               ncols: int = 4, norm: bool = True, cmap: str = None) -> Figure:
    """不使用torchvision的make_grid函数"""
    # 功能: 不适用torchvision的make_grid函数, 纯plt的make_grid函数.

    # cmap: viridis(None), "gray", "gray_r", "hot",
    ###
    # Tensor: [N, C, H, W]. 0-1
    # ndarray: [N, H, W, C]. 0-1
    # 将ax传入前, 先进行config_ax
    if images.ndim != 4:
        raise ValueError(f"images.ndim: {images.ndim}")
    if isinstance(images, ndarray):
        images = torch.from_numpy(images)
        images = images.permute(0, 3, 1, 2)
    n = images.shape[0]
    nrows = int(math.ceil(n / ncols))
    if fig is None:
        fig = plt.figure(figsize=(2 * ncols, 2 * nrows), dpi=200)
    axs = fig.subplots(nrows, ncols)
    #
    if norm:
        images = normalize(images)
    images = images.permute(0, 2, 3, 1)
    images = images.detach().cpu().numpy()
    #
    for i in range(nrows):
        axs_r = (axs[i] if nrows > 1 else axs)
        for j in range(ncols):
            idx = i * ncols + j
            ax: Axes = (axs_r[j] if ncols > 1 else axs_r)
            ax.axis("off")
            if idx >= n:
                continue
            # 若未指定vim, vmax, 则vmin=images.min(), vmax=images.max()
            # 随后将[vmin, vmax] norm-> [0, 1]
            image = images[idx]
            ax.imshow(image, cmap=cmap, origin="upper", vmin=0, vmax=1)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    return fig


# if __name__ == "__main__":
#     import os
#     import torchvision.transforms as tvt
#     DATASETS_PATH = os.environ.get("DATASETS_PATH")
#     assert DATASETS_PATH is not None
#     from torchvision.datasets import FashionMNIST
#     transform = tvt.Compose(
#         [tvt.transforms.ToTensor(), tvt.Normalize((0.5,), (0.5,))])
#     train_dataset = FashionMNIST(
#         root=DATASETS_PATH, train=True, transform=transform, download=True)
#     data = [train_dataset[i][0] for i in range(22)]  # x,y 中取x
#     print(data[0].shape)  # [1, 28, 28]
#     fig, ax = plt.subplots(figsize=(8, 8), dpi=200)
#     make_grid(torch.stack(data), ax)
#     plt.show()

#     #
#     make_grid2(torch.stack(data), cmap='gray')
#     plt.show()


def config_ax(ax: Axes, title: str = None, xlabel: str = None, ylabel: str = None,
              xlim: Tuple[float, float] = None,
              ylim: Tuple[float, float] = None,
              xticks: List[float] = None, xticks_labels: List[str] = None,
              yticks: List[float] = None, yticks_labels: List[str] = None,
              xticks_rotate90=False) -> None:
    if title is not None:
        ax.set_title(title, fontsize=20)
    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=8, fontsize=16)
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=8, fontsize=16)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xticks is not None:
        ax.set_xticks(xticks, xticks_labels)
    if yticks is not None:
        ax.set_yticks(yticks, yticks_labels)

    ax.tick_params(axis="both", which="major", labelsize=14)
    #
    if xticks_rotate90:
        ax.tick_params(axis='x', which='major', rotation=90)


Item = Dict[str, float]  # e.g. step, loss


def read_tensorboard_file(fpath: str) -> Dict[str, List[Item]]:
    """读取fpath中的scalars信息. 变为"""
    ea = EventAccumulator(fpath)
    ea.Reload()
    res = {}
    tags = ea.Tags()['scalars']
    for tag in tags:
        values = ea.Scalars(tag)
        _res = []
        for v in values:
            _res.append({"step": v.step, "value": v.value})
        res[tag] = _res
    return res


def tensorboard_smoothing(values: List[float], smooth: float = 0.9) -> List[float]:
    """不需要传入step"""
    # [0.81 0.9 1]. res[2] = (0.81 * values[0] + 0.9 * values[1] + values[2]) / 2.71
    norm_factor = 1
    x = 0
    res = []
    for i in range(len(values)):
        x = x * smooth + values[i]  # 指数衰减
        res.append(x / norm_factor)
        #
        norm_factor *= smooth
        norm_factor += 1
    return res


# if __name__ == "__main__":
#     fpath = "/home/jintao/Desktop/coding/python/ml_alg/asset/events.out.tfevents.1658302059.jintao.13896.0"
#     loss = read_tensorboard_file(fpath)["train_loss"]
#     v = [l["value"] for l in loss]
#     step = [l["step"] for l in loss]
#     sv = tensorboard_smoothing(v, 0.9)
#     print(sv[490//5 - 1], v[490//5-1])

#     def plot_loss():
#         fig, ax = plt.subplots(figsize=(10, 5))
#         cg, cb = "#FFE2D9", "#FF7043"
#         config_ax(ax, title="Plot_Loss", xlabel="Epoch", ylabel="Loss")
#         ax.plot(step, v, color=cg)
#         ax.plot(step, sv, color=cb)
#     plot_loss()
#     plt.savefig("runs/images/1.png", dpi=200, bbox_inches='tight')
#     # plt.show()
