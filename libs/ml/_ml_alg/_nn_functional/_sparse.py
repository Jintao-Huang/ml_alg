# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ...._types import *
# from libs import *


# def _one_hot(x: Tensor, n_classes: int = -1) -> Tensor:
#     """
#     x: Tensor[long]. [N]
#     """
#     if n_classes == -1:
#         n_classes = x.max() + 1
#     return torch.eye(n_classes, dtype=torch.long, device=x.device)[x]

def one_hot(x: Tensor, n_classes: int = -1) -> Tensor:
    """
    x: Tensor[long]. [N]
    return: Tensor[long]. [N, n_classes]
    """
    if n_classes == -1:
        n_classes = int(x.max().item()) + 1
    res = torch.zeros((x.shape[0], n_classes),
                      dtype=torch.long, device=x.device)
    res[torch.arange(x.shape[0]), x] = 1
    return res


# if __name__ == "__main__":
#     x = torch.randint(0, 10, (1000,))  # long
#     # y = ml.test_time(lambda: _one_hot(x), number=10)
#     y2 = ml.test_time(lambda: one_hot(x), number=10)
#     y3 = ml.test_time(lambda: F.one_hot(x), number=10)
#     # print(torch.allclose(y, y2))
#     print(torch.allclose(y2, y3))



def embedding(x: Tensor, weight: Tensor, padding_idx: int) -> Tensor:
    N = x.shape[0]
    E = weight.shape[1]
    res = torch.empty(N, E)
    mask = x != padding_idx
    res[mask] = weight[x[mask]]
    with torch.no_grad():
        res[~mask] = weight[x[~mask]]  # no grad.
    return res


# if __name__ == "__main__":
#     weight = torch.randn(100, 512, requires_grad=True)
#     x = torch.arange(20, dtype=torch.long)
#     y = F.embedding(x, weight, 1)
#     y2 = embedding(x, weight, 1)
#     print(torch.allclose(y, y2))
#     #
#     y.mean().backward()
#     g = weight.grad
#     weight.grad = None
#     y2.mean().backward()
#     g2 = weight.grad
#     print(torch.allclose(g, g2))
