import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def visualize_sample(data: ndarray, labels: ndarray, ax: Axes) -> None:
    """data可以先降维后输入"""
    # data是2D的[n,f]的ndarray, f=2. float
    # labels: [n]. int
    data_0 = data[labels == 0]
    data_1 = data[labels == 1]
    del data
    #
    ax.scatter(data_0[:, 0], data_0[:, 1], edgecolor="#333", label="Class 0")
    ax.scatter(data_1[:, 0], data_1[:, 1], edgecolor="#333", label="Class 0")
    ax.set_title("Dataset samples")
    ax.set_xlabel("$x_2$")
    ax.set_ylabel("$x_1$")
    ax.legend()


if __name__ == "__main__":
    try:
        from .datasets import XORDataset
    except ImportError:
        from datasets import XORDataset
    dataset = XORDataset()
    fig, ax = plt.subplots(figsize=(4, 4))
    visualize_sample(dataset.data.numpy(), dataset.labels.numpy(), ax)
    plt.show()
