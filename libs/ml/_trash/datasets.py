# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


from torch.utils.data import Dataset
import torch
from torch import Tensor
from typing import Tuple

__all__ = ["XORDataset"]

class XORDataset(Dataset):
    # examples
    def __init__(self, n_samples: int = 256, std: float = 0.1) -> None:
        super(XORDataset, self).__init__()
        self.n_samples = n_samples
        self.std = std
        self.data, self.labels = self._generate_xor()

    def _generate_xor(self) -> Tuple[Tensor, Tensor]:
        data = torch.randint(0, 2, size=(self.n_samples, 2), dtype=torch.long)
        labels = torch.bitwise_xor(data[:, 0], data[:, 1])
        data = data.float()
        data += torch.randn(self.n_samples, 2) * self.std  # 3std 定理
        return data, labels

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.data[idx], self.labels[idx]

    def __len__(self) -> int:
        return self.n_samples
        

# if __name__ == "__main__":
#     dataset = XORDataset(200, 0.1)
#     data, labels = dataset.data, dataset.labels
#     print(data.shape, labels.shape)
#     print(data[:10], labels[:10])
