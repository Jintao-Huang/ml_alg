# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from typing import Union, Optional, Tuple
from torch import Tensor
import torch
__all__ = ["accuracy_score", "confusion_matrix", "precision_recall_fscore",
           "precision_score", "recall_score", "f1_score", "fbeta_score",
           "euclidean_distances", "mean_squared_error",
           "precision_recall_curve", "average_precision_score", "roc_curve", "roc_auc_score"]

# y_pred在前, 保持与torch的loss一致

"""
tp: t=1,p=1. 或t=p=i或sum_i{t=p=i}
fp: t=0, p=1
tn: t=0, p=0
fn: t=1, p=0
ti: t=i或sum_i{t=i}
pi: p=i或sum_i{t=i}

#
下面的指标都是在[0..1]之间
acc = (tp+tn) / (tp+tn+fp+fn)
prec = (tp) / (tp+fp)
  prec=(t=1,p=1) / (p=1)
recall = (tp) / (tp+fn)
  recall=(t=1,p=1) / (t=1)
f1 = (2 * prec * recall) / (prec + recall)
  2/f1 = 1/prec + 1/recall
f_beta = (1+beta^2)*prec*recall/ ((beta^2)*prec + recall)
  (1+beta^2)/f1 = 1/prec + beta^2/recall
"""


def accuracy_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """
    y_pred: Tensor[long]
    y_true: Tensor[long]
    """
    return torch.count_nonzero(y_true == y_pred) / y_pred.shape[0]


def confusion_matrix(y_pred: Tensor, y_true: Tensor,
                     normalize: Optional[str] = None) -> Tensor:
    """
    y_pred: Tensor[long]
    y_true: Tensor[long]
    normalize: 计算完count后, 对哪个维度进行归一化. {"true", "pred", "all", None}. 一般可以对"true"归一化
        "true", "pred"可能出现0/0的情况, 我们用0表示.
    """
    # 横向: true; 竖向: pred. e.g. c[1,2]:t=1,p=2
    n_labels = int(y_true.max().item()) + 1
    # 遍历y_pred, y_true. 每次cm[t][p] += 1
    #   使用向量化技巧: y_pred, y_true -> x, 然后对x进行计数.
    x = y_true * n_labels + y_pred
    cm = x.bincount(minlength=n_labels)
    cm = cm.reshape(n_labels, n_labels)
    # 归一化
    if normalize is None:
        return cm
    if normalize == "true":
        cm /= cm.sum(dim=1, keepdim=True)
    elif normalize == "pred":
        cm /= cm.sum(dim=0, keepdim=True)
    elif normalize == "all":
        cm /= cm.sum()
    else:
        raise ValueError(f"normalize: {normalize}")
    cm.nan_to_num_(nan=0)  # 除0, 则用0表示
    return cm


def _precision(tp: Tensor, pi: Tensor) -> Tensor:
    prec = tp / pi
    return prec.nan_to_num_(nan=0.)


def _recall(tp: Tensor, ti: Tensor) -> Tensor:
    recall = tp / ti
    return recall.nan_to_num_(nan=0.)


def _fscore(prec: Tensor, recall: Tensor, beta: float = 1.) -> Tensor:
    beta2 = beta**2
    fscore = (1+beta2)*prec*recall / (beta2*prec + recall)
    return fscore.nan_to_num_(nan=0.)


def precision_recall_fscore(y_pred: Tensor, y_true: Tensor,
                            beta: float = 1., average: Optional[str] = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    y_pred: Tensor[long]
    y_true: Tensor[long]
    beta: [0, +inf), beta越大, recall的权重越大. (beta=0, 则不考虑recall)
    average: {'micro', 'macro', None}. 'binary'可以通过None实现.
    return: prec, recall, fscore
    """
    # micro: 先平均(总的tp, 总的fp), 然后计算metrics(prec, recall, fscore)
    # macro: 先计算各个的metrics, 再求平均
    if average == "micro":
        # 类似于accuracy: (prec=recall=fscore=accuracy)
        tp = torch.count_nonzero(y_true == y_pred)  # sum{t=p=i}
        t = p = y_pred.shape[0]  # sum{t=i}, sum{p=i}
        prec, recall = _precision(tp, p), _recall(tp, t)
        fscore = _fscore(prec, recall, beta)
        return prec, recall, fscore
    cm = confusion_matrix(y_pred, y_true, normalize=None)
    tp = cm.diag()  # t=p=i
    ti = cm.sum(dim=1)  # t=i
    pi = cm.sum(dim=0)  # p=i
    prec, recall = _precision(tp, pi), _recall(tp, ti)
    fscore = _fscore(prec, recall, beta)
    if average == None:
        return prec, recall, fscore
    elif average == "macro":
        return prec.mean(), recall.mean(), fscore.mean()
    else:
        raise ValueError(f"average: {average}")


def precision_score(y_pred: Tensor, y_true: Tensor, average: Optional[str] = None) -> Tensor:
    return precision_recall_fscore(y_pred, y_true, 1., average)[0]


def recall_score(y_pred: Tensor, y_true: Tensor, average: Optional[str] = None) -> Tensor:
    return precision_recall_fscore(y_pred, y_true, 1., average)[1]


def fbeta_score(y_pred: Tensor, y_true: Tensor,
                beta: float = 1., average: Optional[str] = None) -> Tensor:
    return precision_recall_fscore(y_pred, y_true, beta, average)[2]


def f1_score(y_pred: Tensor, y_true: Tensor, average: Optional[str] = None) -> Tensor:
    return fbeta_score(y_pred, y_true, 1., average)


# if __name__ == "__main__":
#     y_true = torch.tensor([0, 1, 3, 3, 1], device='cuda')
#     y_pred = torch.tensor([0, 2, 1, 3, 1], device='cuda')
#     print(accuracy_score(y_pred, y_true))
#     print(confusion_matrix(y_pred, y_true))
#     print(precision_recall_fscore(y_pred, y_true))
#     print(precision_recall_fscore(y_pred, y_true, average="micro"))
#     print(precision_recall_fscore(y_pred, y_true, average="macro"))
#     from sklearn.metrics import precision_recall_fscore_support, confusion_matrix as _confusion_matrix
#     y_true = y_true.cpu().numpy()
#     y_pred = y_pred.cpu().numpy()
#     print(_confusion_matrix(y_true, y_pred))
#     print(precision_recall_fscore_support(y_true, y_pred, zero_division=0))
#     print(precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0))
#     print(precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0))


def euclidean_distances(X: Tensor, Y: Tensor, XX: Optional[Tensor] = None, YY: Optional[Tensor] = None,
                        squared: bool = False) -> Tensor:
    """
    X: shape[N1, F]. Tensor[float]
    Y: shape[N2, F]. Tensor[float]
    XX: ij,ij->i. 对F做内积, N1逐位运算. shape[N1]
    YY: ij,ij->i. shape[N1]
    return: shape[N1, N2]
    """
    # dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y)). x, y: [F]
    #   (x-y)^2
    if XX is None:
        XX = torch.einsum("ij,ij->i", X, X)
    if YY is None:
        YY = torch.einsum("ij,ij->i", Y, Y)
    # 减少分配空间导致的效率下降
    res = X @ Y.T
    res.mul_(-2).add_(XX[:, None]).add_(YY[None, :])
    res.clamp_min_(0.)  # 避免sqrt{负数}
    return res if squared else res.sqrt_()


# if __name__ == "__main__":
#     # test einsum
#     x = torch.randn(100)
#     print(torch.einsum("i,i->", x, x))

if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *

# if __name__ == "__main__":
#     # test einsum 的speed
#     X = torch.randn(2000, 2000)
#     Y = X.T.contiguous()
#     a = libs_ml.test_time(lambda: X@Y)
#     a = libs_ml.test_time(lambda: X@X.T)
#     Y = X.T.contiguous()
#     b = libs_ml.test_time(lambda: torch.einsum("ij,ij->i", X, Y))
#     c = libs_ml.test_time(lambda: torch.einsum("ij,ji->i", X, X))  # 慢!
#     print(torch.allclose(b, c, rtol=1e-4, atol=1e-4))
#     #
#     print()
#     libs_ml.test_time(lambda: torch.einsum("ij,ij->ij", X, Y))
#     libs_ml.test_time(lambda: X ** 2)  # 慢!
#     libs_ml.test_time(lambda: X * X)

# if __name__ == "__main__":
#     # test inplace. 速度类似. 见max
#     X = torch.randn(2000, 2000)
#     libs_ml.test_time(lambda: torch.sqrt(X), number=1)
#     libs_ml.test_time(lambda: torch.sqrt_(X), number=1)


# if __name__ == "__main__":
#     libs_ml.test_time(lambda: torch.zeros((2000, 2000)))
#     libs_ml.test_time(lambda: torch.zeros((2000, 2000)))


# if __name__ == "__main__":
#     x = torch.randn(1000, 2000)
#     y = torch.randn(4000, 2000)
#     x_np = x.numpy()
#     y_np = y.numpy()
#     from sklearn.metrics import euclidean_distances as _euclidean_distances
#     a = libs_ml.test_time(lambda: euclidean_distances(x, y), number=20)
#     b = libs_ml.test_time(lambda: _euclidean_distances(x_np, y_np), number=20)  # 慢!
#     print(a, b)
#     print(torch.allclose(a, torch.from_numpy(b), rtol=1e-4, atol=1e-4))


def mean_squared_error(y_pred: Tensor, y_true: Tensor, *, reduction="mean",
                       squared: bool = True) -> Tensor:
    """
    y_pred: [N, F]. Tensor[float]
    y_true: [N, F]. Tensor[float]
    reduction: {"mean", "sum", "none"}
    return: [N, F] or []
    """
    # torch.mean((y_pred - y_true) ** 2, dim=0)
    res = y_true - y_pred
    if reduction == "none":
        res.mul_(res)
        return res if squared else res.sqrt_()
    #
    res = torch.einsum("ij,ij->", res, res)
    if reduction == "mean":
        res.div_(y_pred.numel())
    elif reduction == "sum":
        pass
    else:
        raise ValueError(f"reduction: {reduction}")
    return res if squared else res.sqrt_()


# if __name__ == "__main__":
#     from sklearn.metrics import mean_squared_error as _mean_squared_error
#     x = torch.randn(1000, 2000)
#     y = torch.randn(1000, 2000)
#     x_np = x.numpy()
#     y_np = y.numpy()
#     a = libs_ml.test_time(lambda: torch.nn.MSELoss(
#         reduction="none")(x, y), number=10)
#     b = libs_ml.test_time(lambda: torch.nn.MSELoss(
#         reduction="mean")(x, y), number=10)
#     c = libs_ml.test_time(lambda: _mean_squared_error(  # 慢!
#         y_np, x_np, multioutput="raw_values"), number=10)
#     d = libs_ml.test_time(lambda: _mean_squared_error(  # 慢!
#         y_np, x_np, multioutput="uniform_average"), number=10)
#     e = libs_ml.test_time(
#         lambda: mean_squared_error(x, y, reduction="none"), number=10)
#     f = libs_ml.test_time(
#         lambda: mean_squared_error(x, y, reduction="mean"), number=10)
#     print(torch.allclose(a, e, atol=1e-6))
#     print(torch.allclose(b, f))


# if __name__ == "__main__":
#     # a = libs_ml.test_time(lambda:  torch.randn(1000, 2000))
#     # a = libs_ml.test_time(lambda:  torch.zeros(1000, 2000))  # 慢20倍
#     # a = libs_ml.test_time(lambda:  torch.empty(1000, 2000))
#     a = torch.zeros(1000, 2000)
#     # libs_ml.test_time(lambda:  a.add_(100))
#     # libs_ml.test_time(lambda:  a + 100)  # 慢2倍
#     x = libs_ml.test_time(lambda:  torch.empty(1000, 2000))
#     libs_ml.test_time(lambda:  x.zero_())
#     libs_ml.test_time(lambda:  x.add_(10.2))
#     libs_ml.test_time(lambda:  x.mul_(10.2))
#     libs_ml.test_time(lambda:  x.div_(10.123))


#

def _calculate_tps_fps(y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """返回tps, fps, threshold. y_pred可以未排序. (只适用于2分类任务)
    return: tps, fps, threshold. tps: p=1,t=1. fps: p=1,t=0
    """
    # 以y_pred的某一个数为threshold时, {该数}以及左边的数为p=1, 右边的数为p=0
    ###
    # sort
    y_pred, idxs = torch.sort(y_pred, descending=True)
    y_true = y_true[idxs]
    # 计算threshold
    _t = torch.tensor([1e-10], dtype=y_pred.dtype, device=y_pred.device)
    _diff = torch.diff(y_pred, append=_t)
    threshold_idx = torch.nonzero(_diff, as_tuple=True)[0]
    tps = torch.cumsum(y_true, -1)[threshold_idx]  # 含当前.
    _1 = torch.tensor([1], dtype=y_pred.dtype, device=y_pred.device)
    fps = (threshold_idx + _1).sub_(tps)
    return tps, fps, y_pred[threshold_idx]  # t1=tps[-1], t0=fps[-1]


def precision_recall_curve(y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """此实现只适用于2分类任务. X轴为R, Y轴为P(R=xi时的最大的P)
    y_pred: [N]. Tensor[float]. scores. {可以未降序排序}
    y_true: [N]. Tensor[Number]
    return: precision, recall. threshold. threshold为从大到小排序. recall为从小到大.
    """
    # 将y_pred进行排序. 从高到低. 然后去重后获得很多threshold. 对每一个threshold进行p, r的计算.
    # 获得很多个(r, p)的坐标点. 将坐标点连成线即为pr曲线.
    ###
    # 要计算每一个threshold的p,r值, 首先要计算每一个threshold的tps(p=1,t=1), p1, t1.
    # 其中: r随着threshold的降低, 会越来越大(单调递增)
    ###
    # tps + fps = pi
    # ti = tps[-1]
    ###

    #
    tps, fps, threshold = _calculate_tps_fps(y_pred, y_true)
    t1 = tps[-1]
    p1 = tps + fps  # threshold_idx + 1
    precision = _precision(tps, p1)
    recall = _recall(tps, t1)
    return precision, recall, threshold


def average_precision_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """此实现只适用于2分类任务. AP的使用只能使用二分类. 可以使用mAP用于多分类. 
    y_pred: [N]. Tensor[float]. scores. {可以未降序排序}
    y_true: [N]. Tensor[Number]
    """
    precision, recall, _ = precision_recall_curve(y_pred, y_true)
    # 计算面积. 计算: [x1,x2;y1,y2]的面积: 其中x1<x2.
    _0 = torch.tensor([0.], dtype=recall.dtype, device=recall.device)
    dx = torch.diff(recall, prepend=_0)
    return dx.mul_(precision).sum()


# if __name__ == "__main__":
#     from sklearn.metrics import precision_recall_curve as _precision_recall_curve, \
#         average_precision_score as _average_precision_score
#     y_pred = torch.rand(1000)
#     y_true = torch.randint(0, 2, (1000,)).long()
#     b = libs_ml.test_time(
#         lambda: average_precision_score(y_pred, y_true), number=10)
#     b2 = libs_ml.test_time(
#         lambda: _average_precision_score(y_true, y_pred), number=10)
#     print(torch.allclose(b, torch.tensor(b2, dtype=torch.float)))
#     #
#     y_pred = torch.tensor([0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.97])
#     y_true = torch.randint(0, 2, (10,))
#     a = libs_ml.test_time(
#         lambda: precision_recall_curve(y_pred, y_true), number=10)
#     a2 = libs_ml.test_time(
#         lambda: _precision_recall_curve(y_true, y_pred), number=10)

#     print(a)
#     print(a2)


if __name__ == "__main__":

    y_pred = torch.rand(10)
    y_true = torch.randint(0, 2, (10,))
    p, r, _ = precision_recall_curve(y_pred, y_true)

    def plot_pr_curve(p, r, fpath):
        fig, ax = plt.subplots(figsize=(10, 8))
        libs_ml.config_ax(ax, title="PR Curve", xlabel="R",
                          ylabel="P", xlim=(0, 1), ylim=(0, 1))
        ax.plot(r, p)
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
    plot_pr_curve(p, r, "runs/images/2.png")
    from sklearn.metrics import precision_recall_curve as _precision_recall_curve
    p, r, _ = _precision_recall_curve(y_true, y_pred)
    plot_pr_curve(p, r, "runs/images/3.png")


def roc_curve(y_pred: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """此实现只适用于2分类任务. X轴为fpr, Y轴为tpr
    y_pred: shape[N], Tensor[float]. {可以未排序}
    y_true: shape[N], Tensor[Number] 
    return: tpr, fpr, threshold. threshold单调递减, tpr, fpr单调递增
    """
    # 将y_pred按降序排列, 然后计算tps, fps, threshold.
    # tpr=r=tp/t1. fpr=fp/t0
    # t1=tps[-1], t0=fps[-1]
    tps, fps, threshold = _calculate_tps_fps(y_pred, y_true)
    t1 = tps[-1]
    t0 = fps[-1]
    #
    tpr = tps / t1
    fpr = fps / t0
    return tpr, fpr, threshold


def roc_auc_score(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """此实现只适用于2分类任务. X轴为fpr, Y轴为tpr
    y_pred: shape[N], Tensor[float]. {可以未排序}
    y_true: shape[N], Tensor[Number] 
    """
    tpr, fpr, _ = roc_curve(y_pred, y_true)
    # 计算面积. 计算: [x1,x2;y1,y2]的面积: 其中x1<x2.
    _0 = torch.tensor([0.], dtype=tpr.dtype, device=tpr.device)
    dx = torch.diff(fpr, prepend=_0)
    return dx.mul_(tpr).sum()


# if __name__ == "__main__":
#     from sklearn.metrics import roc_auc_score as _roc_auc_score, roc_curve as _roc_curve
#     # y_pred = torch.rand(1000)
#     # y_true = torch.randint(0, 2, (1000,)).long()
#     # a = libs_ml.test_time(
#     #     lambda: roc_auc_score(y_pred, y_true), number=10)
#     # a2 = libs_ml.test_time(
#     #     lambda: _roc_auc_score(y_true, y_pred), number=10)
#     # print(torch.allclose(a, torch.tensor(a2, dtype=torch.float)))

#     #
#     y_pred = torch.tensor([0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.97])
#     y_true = torch.randint(0, 2, (10,))

#     b = libs_ml.test_time(
#         lambda: roc_curve(y_pred, y_true), number=10)
#     b2 = libs_ml.test_time(
#         lambda: _roc_curve(y_true, y_pred), number=10)
#     print(b)
#     print(b2)
#     #

#     def plot_roc_curve(tpr, fpr, fpath):
#         fig, ax = plt.subplots(figsize=(10, 8))
#         libs_ml.config_ax(ax, title="ROC Curve", xlabel="FPR",
#                           ylabel="TPR", xlim=(0, 1), ylim=(0, 1))
#         ax.plot(r, p)
#         plt.savefig(fpath, dpi=200, bbox_inches='tight')
#     tpr, fpr, _ = roc_curve(y_pred, y_true)
#     plot_roc_curve(tpr, fpr, "runs/images/4.png")
#     tpr, fpr, _ = _roc_curve(y_true, y_pred)
#     plot_roc_curve(tpr, fpr, "runs/images/5.png")
