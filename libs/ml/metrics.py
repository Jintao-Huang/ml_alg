# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:
from typing import Union
from torch import Tensor
import torch
__all__ = ["accuracy"]


def accuracy(y_pred: Tensor, y_true: Tensor,
             return_count: bool = False) -> Tensor:
    """y_pred在前, 保持与torch的loss一致
        y_pred: [N]
        y_true: [N]
        return_count: 返回count. 否则: 返回百分比
    """
    n_samples = y_true.shape[0]
    res = torch.count_nonzero(y_true == y_pred)
    return res if return_count else res / n_samples


if __name__ == "__main__":
    y_true = torch.tensor([1, 2, 3])
    y_pred = torch.tensor([2, 1, 3])
    print(accuracy(y_true, y_pred))
    print(accuracy(y_true, y_pred, True))
