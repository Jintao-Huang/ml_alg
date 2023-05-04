# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ...._types import *
# from libs import *


def nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: Tensor[float]. [N, F]
    target: Tensor[long]. [N]
    return: []
    """
    N, n_labels = pred.shape[:2]
    target = F.one_hot(target, n_labels)  # long
    res = pred.mul(target)
    return -res.sum() / N


# if __name__ == "__main__":
#     x = torch.randn((1000, 10))
#     x2 = torch.randint(0, 10, (1000,))
#     y = ml.test_time(lambda: nll_loss(x, x2))
#     y2 = ml.test_time(lambda: F.nll_loss(x, x2))
#     print(y, y2)
#     print(torch.allclose(y, y2))


def cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: Tensor[float]. [N, F]
    target: Tensor[long]. [N]
    return: []
    """

    return F.nll_loss(F.log_softmax(pred, dim=1), target)

# if __name__ == "__main__":
#     x = torch.randn((1000, 10))
#     x2 = torch.randint(0, 10, (1000,))
#     y = ml.test_time(lambda: cross_entropy(x, x2))
#     y2 = ml.test_time(lambda: F.cross_entropy(x, x2))
#     print(torch.allclose(y, y2))


def label_smoothing_cross_entropy(pred: Tensor, target: Tensor,
                                  smoothing: float = 0.01) -> Tensor:
    """
    pred: [N, F]. Tensor[float]. 未过softmax
    target: [N]. Tensor[long]
    smoothing: 若smoothing为0.1, 则target=4, n_labels=5, 对应:
        [0.02, 0.02, 0.02, 0.02, 0.92]
        此时: pred为: [0.02, 0.02, 0.02, 0.02, 0.92]时损失最小. 可以通过求导=0得出. 
    return: []
    """
    n_labels = pred.shape[1]
    # 构造target. 将target->[N, F]. ，target[i]的第target和样本设为1-smoothing.
    # 然后加上smoothing / n_labels
    res: Tensor = F.one_hot(target, n_labels)  # long
    res = res * (1-smoothing)
    res.add_(smoothing / n_labels)
    # 计算loss
    res.mul_(F.log_softmax(pred, dim=-1))
    return -res.sum() / pred.shape[0]


# if __name__ == "__main__":
#     x = torch.randn((1000, 100))
#     x2 = torch.randint(0, 100, (1000,))
#     y = ml.test_time(lambda: F.cross_entropy(x, x2), number=10)
#     y1 = ml.test_time(lambda: F.cross_entropy(x, x2, label_smoothing=0.1), number=10)
#     y2 = ml.test_time(lambda: label_smoothing_cross_entropy(x, x2, smoothing=0.1), number=10)
#     print(y, y1, y2)


def binary_cross_entropy_with_logits(pred: Tensor, target: Tensor) -> Tensor:
    """binary_cross_entropy数值不稳定. 
    pred: Tensor[float]. [N]
    target: Tensor[float]. [N]
    return: []
    """
    # -logsigmoid(x)*target-logsigmoid(-x)*(1-target)
    # logsigmoid(-x)) == log(1 - sigmoid(x))
    ###
    p_sig: Tensor = F.logsigmoid(pred)
    pm_sig: Tensor = F.logsigmoid(-pred)
    res = p_sig.mul_(target)
    res.add_(pm_sig.mul_((1-target)))
    return -res.mean()


# if __name__ == "__main__":
#     x = torch.randn((1000,))
#     x2 = torch.randint(0, 2, (1000,), dtype=torch.float)
#     y = ml.test_time(
#         lambda: binary_cross_entropy_with_logits(x, x2), number=100)
#     y2 = ml.test_time(
#         lambda: F.binary_cross_entropy_with_logits(x, x2), number=100)
#     print(torch.allclose(y, y2))


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """
    pred: [N, F]. Tensor[float]
    target: [N, F]. Tensor[float]
    return: []
    """
    # torch.mean((y_pred - y_true) ** 2, dim=0)
    diff = target.sub(pred)
    res = torch.einsum("ij,ij->", diff, diff)
    return res.div_(pred.numel())

# if __name__ == "__main__":
#     x = torch.randn((10000, 1000))
#     x2 = torch.randn((10000, 1000))
#     y = ml.test_time(lambda: mse_loss(x, x2), number=10)
#     y2 = ml.test_time(lambda: F.mse_loss(x, x2), number=10)
#     print(torch.allclose(y, y2))


def smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.) -> Tensor:
    """diff=beta为loss1, loss2的分界线."""
    # diff=beta处, loss1, loss2的导数值(=1), 值相等(=beta/2). loss(diff=0)=0
    cond = target.sub(pred).abs_().lt(beta)
    loss1 = F.mse_loss(pred, target, reduction="none").div_(2 * beta)
    loss2 = F.l1_loss(pred, target, reduction="none").sub_(beta / 2)
    return torch.where(cond, loss1, loss2).mean()


# if __name__ == "__main__":
#     x = torch.randn(100000)
#     y = torch.randn(100000)
#     y1 = ml.test_time(lambda: F.smooth_l1_loss(x, y, beta=2), number=10, warmup=1)
#     y2 = ml.test_time(lambda: smooth_l1_loss(x, y, beta=2), number=10)
#     print(torch.allclose(y1, y2))


def cosine_embedding_loss(x1: Tensor, x2: Tensor, target: Tensor, 
                          margin: float = 0.)->Tensor:
    """
    x1: [N, D]
    x2: [N, D]
    target: [N]
    margin: -1~1. 建议 0~0.5. 
    return: []
    """
    cos_sim = F.cosine_similarity(x1, x2)
    loss1 = 1 - cos_sim  # target == 1
    loss2 = (cos_sim - margin).clamp_min_(0)  # target == -1
    return torch.where(target == 1, loss1, loss2).mean()

# if __name__ == "__main__":
#     x1 = torch.randn(1000, 768)
#     x2 = torch.randn(1000, 768)
#     target = torch.where(torch.randn(1000) >= 0, 1, -1)
#     y = ml.test_time(lambda: cosine_embedding_loss(x1, x2, target, 0.5), 10)
#     y2 = ml.test_time(lambda: F.cosine_embedding_loss(x1, x2, target, 0.5), 10)
#     print(torch.allclose(y, y2))
