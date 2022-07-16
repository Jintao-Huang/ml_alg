import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from torch import device as Device, Tensor
from torch.nn import Module
import torch
from matplotlib.colors import to_rgb
from typing import Optional, List, Union
from torch.utils.data import DataLoader, TensorDataset
from numpy import ndarray


__all__ = ["plot_classification_map", "visualize_samples"]


@torch.no_grad()
def plot_classification_map(model: Module, device: Device,
                            extent: Tuple[float, float, float, float], n_labels: int, ax: Axes) -> None:
    """只支持输入为2D的情形"""
    # extent: lrtb. 即: (*xlim, *ylim)
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
    x = x.reshape(-1, 2)
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
    y = y.reshape((*shape[:2], n_labels))
    # y: [NX, NY, NL]. ci: [NL, 3]
    # res: [NX, NY, 3]
    output_image = y @ ci
    output_image = output_image.numpy()

    ax.imshow(output_image, origin="lower", extent=extent)
    ax.grid(False)


def visualize_samples(data: Union[Tensor, ndarray], labels: Union[Tensor, ndarray], ax: Axes) -> None:
    """data可以先降维后输入. 可支持多分类"""
    # data是2D的[n,f]的Tensor, f=2. float
    # labels: [n]. int
    if isinstance(data, Tensor):
        data = data.detach_().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach_().cpu().numpy()
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


if __name__ == "__main__":
    try:
        from . import XORDataset, MLP_L2
    except ImportError:
        from datasets import XORDataset
        from models import MLP_L2

    dataset = XORDataset(200)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=400)
    dataset.labels[:100][dataset.labels[:100] == 1] = 2
    n_labels = 3
    model = MLP_L2(2, 4, n_labels)
    plot_classification_map(model, Device(
        'cuda'), (-0.5, 1.5, 0, 2), n_labels, ax)
    visualize_samples(dataset.data, dataset.labels, ax)
    # plt.savefig("1.png", bbox_inches='tight')
    plt.show()
