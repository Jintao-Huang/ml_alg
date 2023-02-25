# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from numba import njit
from numpy import ndarray
from typing import Optional, Tuple, Literal
from torch import Tensor
import torch.nn.functional as F
from torch import dtype as Dtype
import torch
from ._cy import calc_rank_loop as _calc_rank_loop

__all__ = [
    "accuracy", "confusion_matrix",
    "precision_recall_fbeta", "precision", "recall", "fbeta_score", "f1_score",
    "precision_recall_curve", "average_precision", "roc_curve", "auroc",
    "r2_score",
    "pairwise_cosine_similarity", "pairwise_euclidean_distance",
    "batched_cosine_similarity", "batched_euclidean_distance",
    "batched_mse", "pairwise_mse", "pairwise_equal",
    "kl_divergence",
    "pearson_corrcoef", "spearman_corrcoef", "calc_rank"
]

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
if __name__ == "__main__":
    from libs import *


def accuracy(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,  # or -1
    top_k: int = 1
) -> Tensor:
    """
    y_pred: Tensor[long]. shape[N] or Tensor[float]. shape[N, C]
    y_true: Tensor[long]. shape[N]
    num_classes: only "multiclass"
    top_k: only "multiclass"
    return: shape[]
    """
    N = y_pred.shape[0]
    if task == "binary":
        assert y_pred.ndim == 1 and top_k == 1
        assert y_pred.dtype not in {torch.float32, torch.float64}  # 传入的使用处理好 threshold.
    elif task == "multiclass":
        assert num_classes is not None
        if num_classes == -1:
            num_classes = int(y_true.max().item()) + 1
        #
        if y_pred.ndim == 2:
            assert num_classes == y_pred.shape[1]
            if top_k > 1:
                y_pred = y_pred.topk(top_k, dim=-1)[1]
                y_true = y_true[:, None]
            else:
                y_pred = y_pred.argmax(dim=1)
    else:
        raise ValueError(f"task: {task}")
    return (y_true == y_pred).count_nonzero() / N


# if __name__ == "__main__":
#     from torchmetrics.functional.classification.accuracy import accuracy as _accuracy
#     preds = torch.randint(0, 10, (1000,))
#     target = torch.randint(0, 10, (1000,))
#     y = libs_ml.test_time(lambda: accuracy(preds, target, "multiclass", num_classes=10), 10)
#     y2 = libs_ml.test_time(lambda: _accuracy(preds, target, "multiclass", num_classes=10), 10)
#     print(y, y2)
#     #
#     from torchmetrics.classification.accuracy import Accuracy
#     acc_metric = Accuracy("multiclass", num_classes=10)
#     acc = libs_ml.test_metric(acc_metric, preds, target)
#     print(acc)


# if __name__ == "__main__":
#     from torchmetrics.functional.classification.accuracy import accuracy as _accuracy
#     target = torch.tensor([0, 2, 2, 1, 1])
#     preds = torch.tensor([[0.9, 0.01, 0.09], [0.5, 0.2, 0.3], [0.4, 0.25, 0.15], [0.3, 0.4, 0.3], [0.1, 0.8, 0.1]])
#     print(libs_ml.test_time(lambda: _accuracy(preds, target, "multiclass", num_classes=3)))
#     print(libs_ml.test_time(lambda: _accuracy(preds, target, "multiclass", num_classes=3, top_k=2)))
#     print(libs_ml.test_time(lambda: accuracy(preds, target, "multiclass", num_classes=3)))
#     print(libs_ml.test_time(lambda: accuracy(preds, target, "multiclass", num_classes=3, top_k=2)))

#     target = torch.tensor([0, 0, 1, 1, 1])
#     preds = torch.tensor([0.1, 0.1, 0.5, 0.9, 0.9])
#     print(libs_ml.test_time(lambda: _accuracy(preds, target, "binary")))
#     print(libs_ml.test_time(lambda: accuracy(preds, target, "binary")))


def confusion_matrix(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,  # -1: "auto"
    normalize: Literal["true", "pred", "all", None] = None
) -> Tensor:
    """
    y_pred: Tensor[long]. shape[N]
    y_true: Tensor[long]. shape[N]
    normalize: 计算完count后, 对哪个维度进行归一化. {"true", "pred", "all", None}. 一般可以对"true"归一化
        "true", "pred"可能出现0/0的情况, 我们用0表示.
    return: shape[N, N]. Tensor[long/float]
    """
    # 横向: true; 竖向: pred. e.g. c[1,2]:t=1,p=2
    if task == "binary":
        num_classes = 2
    elif task == "multiclass":
        assert num_classes is not None
        if num_classes == -1:
            num_classes = int(y_true.max().item()) + 1
    else:
        raise ValueError(f"task: {task}")
    # 遍历y_pred, y_true. 每次cm[t][p] += 1
    #   使用向量化技巧: y_pred, y_true -> x, 然后对x进行计数.
    idx = (y_true * num_classes).add_(y_pred)
    cm = idx.bincount(minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    # 归一化
    if normalize is None:
        return cm
    if normalize == "true":
        cm = cm / cm.sum(dim=1, keepdim=True)
    elif normalize == "pred":
        cm = cm / cm.sum(dim=0, keepdim=True)
    elif normalize == "all":
        cm = cm / cm.sum()
    else:
        raise ValueError(f"normalize: {normalize}")
    cm.nan_to_num_(nan=0)  # 除0, 则用0表示
    return cm


# if __name__ == "__main__":
#     print()
#     from torchmetrics.functional.classification.confusion_matrix import confusion_matrix as _confusion_matrix
#     preds = torch.randint(0, 10, (100,))
#     target = torch.randint(0, 10, (100,))
#     y = libs_ml.test_time(lambda: confusion_matrix(preds, target, "multiclass", num_classes=-1, normalize="true"), 10)
#     y2 = libs_ml.test_time(lambda: _confusion_matrix(
#         preds, target, "multiclass", num_classes=int(target.max().item()) + 1, normalize="true"), 10)
#     print(torch.allclose(y, y2))
#     #
#     preds = torch.randint(0, 2, (100,))
#     target = torch.randint(0, 2, (100,))
#     y = libs_ml.test_time(lambda: confusion_matrix(preds, target, "binary", normalize="true"), 10)
#     y2 = libs_ml.test_time(lambda: _confusion_matrix(preds, target, "binary", normalize="true"), 10)
#     print(torch.allclose(y, y2))


def _precision(tp: Tensor, pi: Tensor) -> Tensor:
    prec = tp / pi
    return prec.nan_to_num_(nan=0.)


def _recall(tp: Tensor, ti: Tensor) -> Tensor:
    recall = tp / ti
    return recall.nan_to_num_(nan=0.)


def _fbeta(prec: Tensor, recall: Tensor, beta: float = 1.) -> Tensor:
    beta2 = beta*beta
    fbeta = (prec * recall).mul_(1+beta2).div_((prec * beta2).add_(recall))
    return fbeta.nan_to_num_(nan=0.)


def precision_recall_fbeta(
    y_pred: Tensor, y_true: Tensor,
    task: Literal["binary", "multiclass"],
    beta: float = 1.,
    #
    num_classes: Optional[int] = None,  # -1: "auto"
    average: Literal["micro", "macro", None] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    y_pred: Tensor[long], [N]
    y_true: Tensor[long], [N]
    beta: [0, +inf), beta越大, recall的权重越大. (beta=0, 则不考虑recall)
    average: {'micro', 'macro', None}. 'binary'可以通过None实现(取index=1即可).
    return: prec, recall, fbeta. shape[], shape[], shape[]
    """
    if task == "binary":
        num_classes = 2
    elif task == "multiclass":
        assert num_classes is not None
        if num_classes == -1:
            num_classes = int(y_true.max().item()) + 1
    else:
        raise ValueError(f"task: {task}")
    # micro: 先平均(总的tp, 总的fp), 然后计算metrics(prec, recall, fbeta)
    # macro: 先计算各个的metrics, 再求平均
    if average == "micro":
        # 类似于accuracy: (prec=recall=fbeta=accuracy)
        tp = torch.count_nonzero(y_true == y_pred)  # sum{t=p=i}
        t = p = y_pred.shape[0]  # sum{t=i}, sum{p=i}
        prec, recall = _precision(tp, p), _recall(tp, t)
        fbeta = _fbeta(prec, recall, beta)
        return prec, recall, fbeta
    cm = confusion_matrix(y_pred, y_true, task, num_classes, normalize=None)
    tp = cm.diag()  # t=p=i
    ti = cm.sum(dim=1)  # t=i
    pi = cm.sum(dim=0)  # p=i
    prec, recall = _precision(tp, pi), _recall(tp, ti)
    fbeta = _fbeta(prec, recall, beta)

    if task == "binary":
        return prec[1], recall[1], fbeta[1]
    #
    if average == None:
        return prec, recall, fbeta
    elif average == "macro":
        return prec.mean(), recall.mean(), fbeta.mean()
    else:
        raise ValueError(f"average: {average}")


def precision(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,
    average: Literal["micro", "macro", None] = None
) -> Tensor:
    return precision_recall_fbeta(y_pred, y_true, task, 1., num_classes, average)[0]


def recall(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,
    average: Literal["micro", "macro", None] = None
) -> Tensor:
    return precision_recall_fbeta(y_pred, y_true, task, 1., num_classes, average)[1]


def fbeta_score(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,
    average: Literal["micro", "macro", None] = None
) -> Tensor:
    return precision_recall_fbeta(y_pred, y_true, task, 1., num_classes, average)[2]


def f1_score(
    y_pred: Tensor,
    y_true: Tensor,
    task: Literal["binary", "multiclass"],
    num_classes: Optional[int] = None,
    average: Literal["micro", "macro", None] = None
) -> Tensor:
    return fbeta_score(y_pred, y_true, task, num_classes, average)


# if __name__ == "__main__":
#     print()
#     from torchmetrics.functional.classification.f_beta import f1_score as _f1_score, fbeta_score as _fbeta_score
#     from torchmetrics.functional.classification.precision_recall import precision as __precision, recall as __recall
#     preds = torch.randint(0, 2, (1000,))
#     target = torch.randint(0, 1, (1000,))
#     target[:10] = 1
#     print(libs_ml.test_time(lambda: _f1_score(preds, target, "binary")))
#     print(libs_ml.test_time(lambda: _fbeta_score(preds, target, "binary", 1.)))
#     print(libs_ml.test_time(lambda: __precision(preds, target, "binary")))
#     print(libs_ml.test_time(lambda: __recall(preds, target, "binary")))
#     print(libs_ml.test_time(lambda: precision_recall_fbeta(preds, target, "binary", 1)))
#     #
#     from torchmetrics.classification.f_beta import F1Score
#     from torchmetrics.classification.precision_recall import Precision, Recall
#     metrics = [
#         Precision("binary"),
#         Recall("binary"),
#         F1Score("binary")
#     ]
#     from libs import *
#     p = libs_ml.test_metric(metrics[0], preds, target)
#     r = libs_ml.test_metric(metrics[1], preds, target)
#     f1 = libs_ml.test_metric(metrics[2], preds, target)
#     print(p, r, f1)

#     #
#     preds = torch.tensor([0, 1, 3, 3, 1], device='cuda')
#     target = torch.tensor([0, 2, 1, 3, 1], device='cuda')
#     num_classes = int(target.max().item()) + 1
#     print(_f1_score(preds, target, "multiclass", num_classes=num_classes, average="macro"))
#     print(_fbeta_score(preds, target, "multiclass", 1., num_classes=num_classes, average="macro"))
#     print(__precision(preds, target, "multiclass", num_classes=num_classes, average="macro"))
#     print(__recall(preds, target, "multiclass", num_classes=num_classes, average="macro"))
#     print(precision_recall_fbeta(preds, target, "multiclass", 1., num_classes, average="macro"))


# if __name__ == "__main__":
#     # test average
#     print()
#     y_true = torch.tensor([0, 1, 3, 3, 1], device='cuda')
#     y_pred = torch.tensor([0, 2, 1, 3, 1], device='cuda')
#     print(accuracy(y_pred, y_true, "multiclass", -1))
#     print(precision_recall_fbeta(y_pred, y_true, "multiclass", num_classes=-1))
#     print(precision_recall_fbeta(y_pred, y_true, "multiclass", num_classes=-1, average="micro"))
#     print(precision_recall_fbeta(y_pred, y_true, "multiclass", num_classes=-1, average="macro"))


def _calculate_tps_fps(y_score: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """返回tps, fps, threshold. y_score可以未排序. (只适用于2分类任务)
    return: tps, fps, threshold. tps: p=1,t=1. fps: p=1,t=0. shape[N], shape[N], shape[N].
    """
    # 以y_score的某一个数为threshold时, {该数}以及左边的数为p=1, 右边的数为p=0
    ###
    # sort
    y_score, idxs = torch.sort(y_score, descending=True)
    y_true = y_true[idxs]
    # 计算threshold
    _t = torch.tensor([1e-10], dtype=y_score.dtype, device=y_score.device)
    _diff = torch.diff(y_score, append=_t)
    threshold_idx = torch.nonzero(_diff, as_tuple=True)[0]
    tps = torch.cumsum(y_true, -1)[threshold_idx]  # 含当前.
    _1 = torch.tensor([1], dtype=y_score.dtype, device=y_score.device)
    fps = (threshold_idx + _1).sub_(tps)
    return tps, fps, y_score[threshold_idx]  # t1=tps[-1], t0=fps[-1]


def precision_recall_curve(y_score: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """(task="binary"). X轴为R, Y轴为P(R=xi时的最大的P)
    y_score: [N]. Tensor[float]. scores. {可以未降序排序}
    y_true: [N]. Tensor[Number]
    return: precision, recall. threshold. threshold为从大到小排序. recall为从小到大. shape[N], shape[N], shape[N].
    # 
    torchmetrics的实现precision_recall_curve中多了个点: R,P,T: (0,1,/). 
        但这个在计算ap时, 这个点不计算, 所以该函数不再加进去.
    """
    # 将y_score进行排序. 从高到低. 然后去重后获得很多threshold. 对每一个threshold进行p, r的计算.
    # 获得很多个(r, p)的坐标点. 将坐标点连成线即为pr曲线.
    ###
    # 要计算每一个threshold的p,r值, 首先要计算每一个threshold的tps(p=1,t=1), p1, t1.
    # 其中: r随着threshold的降低, 会越来越大(单调递增)
    ###
    # tps + fps = pi
    # ti = tps[-1]
    ###
    tps, fps, threshold = _calculate_tps_fps(y_score, y_true)
    t1 = tps[-1]
    p1 = fps.add_(tps)  # threshold_idx + 1
    precision = _precision(tps, p1)
    recall = _recall(tps, t1)
    return precision, recall, threshold


def average_precision(y_score: Tensor, y_true: Tensor) -> Tensor:
    """(task="binary"). AP的使用只能使用二分类. 可以使用mAP用于多分类. 
    y_score: Tensor[float]. shape[N]. scores. {可以未降序排序}
    y_true: Tensor[Number]. shape[N]
    return: Tensor[float]. shape[]
    """
    precision, recall, _ = precision_recall_curve(y_score, y_true)
    # 计算面积. 计算: [x1,x2;y1,y2]的面积: 其中x1<x2.
    _0 = torch.tensor([0.], dtype=recall.dtype, device=recall.device)
    dx = torch.diff(recall, prepend=_0)
    return torch.einsum("i,i->", dx, precision)


# if __name__ == "__main__":
#     from torchmetrics.functional import average_precision as _average_precision, precision_recall_curve as _precision_recall_curve
#     y_score = torch.rand(1000)
#     y_true = torch.randint(0, 2, (1000,)).long()
#     b = libs_ml.test_time(
#         lambda: average_precision(y_score, y_true), number=10)
#     b2 = libs_ml.test_time(
#         lambda: _average_precision(y_score, y_true, "binary"), number=10)
#     print(torch.allclose(b, b2), b)
#     from torchmetrics.classification.average_precision import AveragePrecision
#     ap_metric = AveragePrecision("binary")
#     ap = libs_ml.test_metric(ap_metric, y_score, y_true)
#     print(ap)
#     #
#     from sklearn.metrics import precision_recall_curve as __precision_recall_curve
#     y_score = torch.tensor([0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.97])
#     y_true = torch.randint(0, 2, (10,))
#     a = libs_ml.test_time(
#         lambda: precision_recall_curve(y_score, y_true), number=10)
#     a2 = libs_ml.test_time(
#         lambda: _precision_recall_curve(y_score, y_true, "binary"), number=10)
#     a3 = libs_ml.test_time(
#         lambda: __precision_recall_curve(y_true, y_score), number=10)
#     print(a)
#     print(a2)
#     print(a3)


# if __name__ == "__main__":
#     y_score = torch.rand(10)
#     y_true = torch.randint(0, 2, (10,))
#     p, r, _ = precision_recall_curve(y_score, y_true)

#     def plot_pr_curve(p, r, fpath):
#         fig, ax = plt.subplots(figsize=(10, 8))
#         libs_ml.config_ax(ax, title="PR Curve", xlabel="R",
#                           ylabel="P", xlim=(0, 1), ylim=(0, 1))
#         ax.plot(r, p)
#         plt.savefig(fpath, dpi=200, bbox_inches='tight')
#     plot_pr_curve(p, r, "asset/images/2.png")
#     from torchmetrics.functional import precision_recall_curve as _precision_recall_curve
#     p, r, _ = _precision_recall_curve(y_score, y_true, "binary")
#     plot_pr_curve(p, r, "asset/images/3.png")


def roc_curve(y_score: Tensor, y_true: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """(task="binary"). X轴为fpr, Y轴为tpr
    y_score: shape[N], Tensor[float]. {可以未排序}
    y_true: shape[N], Tensor[Number] 
    return: tpr, fpr, threshold. threshold单调递减, tpr, fpr单调递增. shape[N], shape[N], shape[N]
    # 
    返回中不含0,0点. 该点对auroc的计算无关, 所以不加入. 
    """
    # 将y_score按降序排列, 然后计算tps, fps, threshold.
    # tpr=r=tp/t1. fpr=fp/t0
    # t1=tps[-1], t0=fps[-1]
    tps, fps, threshold = _calculate_tps_fps(y_score, y_true)
    t1 = tps[-1]
    t0 = fps[-1]
    #
    tpr = tps / t1
    fpr = fps / t0
    return tpr, fpr, threshold


def auroc(y_score: Tensor, y_true: Tensor) -> Tensor:
    """(task="binary"). X轴为fpr, Y轴为tpr
    y_score: Tensor[float]. shape[N]. {可以未排序}
    y_true: Tensor[Number]. shape[N]. 
    return: Tensor[float]. shape[]
    """
    tpr, fpr, _ = roc_curve(y_score, y_true)
    # 计算面积. 计算: [x1,x2;y1,y2]的面积: 其中x1<x2.
    _0 = torch.tensor([0.], dtype=tpr.dtype, device=tpr.device)
    dx = torch.diff(fpr, prepend=_0)
    return torch.einsum("i,i->", dx, tpr)


# if __name__ == "__main__":
#     from torchmetrics.functional import auroc as _auroc
#     y_score = torch.rand(1000)
#     y_true = torch.randint(0, 2, (1000,)).long()
#     print(libs_ml.test_time(lambda:auroc(y_score, y_true)))
#     print(libs_ml.test_time(lambda:_auroc(y_score, y_true, "binary")))
#     print()
#     from torchmetrics.classification.auroc import AUROC
#     auroc_metric = AUROC("binary")
#     print(libs_ml.test_metric(auroc_metric, y_score, y_true))
#     #
#     y_score = torch.rand(1000, 10)
#     y_true = torch.randint(0, 2, (1000, 10)).long()
#     print(libs_ml.test_time(lambda:auroc(y_score.flatten(), y_true.flatten())))
#     print(libs_ml.test_time(lambda:_auroc(y_score, y_true, "binary")))

# if __name__ == "__main__":
#     from sklearn.metrics import roc_auc_score as _roc_auc_score, roc_curve as _roc_curve
#     y_score = torch.rand(1000)
#     y_true = torch.randint(0, 2, (1000,)).long()
#     a = libs_ml.test_time(
#         lambda: auroc(y_score, y_true), number=10)
#     a2 = libs_ml.test_time(
#         lambda: _roc_auc_score(y_true, y_score), number=10)
#     print(torch.allclose(a, torch.tensor(a2, dtype=torch.float)))

#     #
#     y_score = torch.tensor([0.1, 0.2, 0.2, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.97])
#     y_true = torch.randint(0, 2, (10,))

#     b = libs_ml.test_time(
#         lambda: roc_curve(y_score, y_true), number=10)
#     b2 = libs_ml.test_time(
#         lambda: _roc_curve(y_true, y_score), number=10)
#     print(b)
#     print(b2)
#     #

#     def plot_roc_curve(tpr, fpr, fpath):
#         fig, ax = plt.subplots(figsize=(10, 8))
#         libs_ml.config_ax(ax, title="ROC Curve", xlabel="FPR",
#                           ylabel="TPR", xlim=(0, 1), ylim=(0, 1))
#         ax.plot(tpr, fpr)
#         plt.savefig(fpath, dpi=200, bbox_inches='tight')
#     tpr, fpr, _ = roc_curve(y_score, y_true)
#     plot_roc_curve(tpr, fpr, "asset/images/4.png")
#     fpr, tpr, _ = _roc_curve(y_true, y_score)
#     plot_roc_curve(tpr, fpr, "asset/images/5.png")


def r2_score(y_pred: Tensor, y_true: Tensor, reduction: Literal["mean", "none"] = "mean") -> Tensor:
    """
    y_pred: shape[N] or [N, F]. Tensor[float].
    y_true: shape[N] or [N, F]. Tensor[float]. 
    return: shape[]. Tensor[float]. 
    """
    # R2 = 1 - u/v. 其中u=MSE(y_true, y_pred). v=Var(y_true)
    # v的方差越大, 可以缓解预测u的误差大.
    # 可以把F理解为batch.
    u = F.mse_loss(y_pred, y_true, reduction="none").mean(dim=0)
    v = torch.var(y_true, dim=0, unbiased=False)  # [F]
    res = u.div_(v).neg_().add_(1)
    if reduction == "mean":
        res = res.mean()
    elif reduction != "none":
        raise ValueError(f"reduction: {reduction}")
    return res  # 1 - u/v


# if __name__ == "__main__":
#     from torchmetrics.functional.regression.r2 import r2_score as _r2_score
#     print()
#     y_pred = torch.randn(10000)
#     y_true = torch.randn(10000)
#     print(libs_ml.test_time(lambda: r2_score(y_pred, y_true)))
#     print(libs_ml.test_time(lambda: _r2_score(y_pred, y_true)))
#     #
#     y_pred = torch.randn(10000, 1000)
#     y_true = torch.randn(10000, 1000)
#     print(libs_ml.test_time(lambda: r2_score(y_pred, y_true)))
#     print(libs_ml.test_time(lambda: _r2_score(y_pred, y_true)))


def batched_cosine_similarity(X: Tensor, Y: Tensor) -> Tensor:
    """或直接使用F.cosine_similarity
    X: [N, F]. float
    Y: [N, F]. float
    return: [N]. float
    """
    res = torch.einsum("ij,ij->i", X, Y)
    res.div_(torch.norm(X, dim=1)).div_(torch.norm(Y, dim=1))
    return res


def pairwise_cosine_similarity(X: Tensor, Y: Tensor) -> Tensor:
    """
    X: shape[N1, F]. Tensor[float]
    Y: shape[N2, F]. Tensor[float]
    return: shape[N1, N2]. Tensor[float]
    """
    # <X, Y> / (|X| |Y|)
    res = X @ Y.T
    X_norm = torch.norm(X, dim=1)  # [N1]
    Y_norm = torch.norm(Y, dim=1)  # [N2]
    res.div_(X_norm[:, None]).div_(Y_norm)
    return res


# if __name__ == "__main__":
#     from torchmetrics.functional.pairwise.cosine import pairwise_cosine_similarity as _pairwise_cosine_similarity
#     x = torch.rand(1000, 1000)
#     y = torch.rand(1000, 1000)
#     z = libs_ml.test_time(lambda: pairwise_cosine_similarity(x, y), 5)
#     z2 = libs_ml.test_time(lambda: _pairwise_cosine_similarity(x, y), 5)
#     print(torch.allclose(z, z2, atol=1e-6))
#     z3 = libs_ml.test_time(lambda: batched_cosine_similarity(x, y), 5)
#     print(torch.allclose(z.ravel()[::1000+1], z3, atol=1e-6))
#     #
#     z4 = libs_ml.test_time(lambda: F.cosine_similarity(x, y), 5)
#     print(torch.allclose(z3, z4, atol=1e-6))
#     z5 = libs_ml.test_time(lambda: F.cosine_similarity(x[:, None], y[None, :], dim=2), 1)
#     print(torch.allclose(z, z5, atol=1e-6))


def batched_euclidean_distance(
    X: Tensor,
    Y: Tensor,
    XX: Optional[Tensor] = None,
    YY: Optional[Tensor] = None,
    squared: bool = False
) -> Tensor:
    """或直接使用F.pairwise_distance
    X: [N, F]. float
    Y: [N, F]. float
    XX: ij,ij->i. [N]
    YY: ij,ij->i. [N]
    return: [N]
    """
    if XX is None:
        XX = torch.einsum("ij,ij->i", X, X)
    if YY is None:
        YY = torch.einsum("ij,ij->i", Y, Y)
    # 减少分配空间导致的效率下降
    res = torch.einsum("ij,ij->i", X, Y)
    res.mul_(-2).add_(XX).add_(YY)
    res.clamp_min_(0.)  # 避免sqrt{负数}
    return res if squared else res.sqrt_()


def pairwise_euclidean_distance(
    X: Tensor,
    Y: Tensor,
    XX: Optional[Tensor] = None,
    YY: Optional[Tensor] = None,
    squared: bool = False
) -> Tensor:
    """
    X: shape[N1, F]. Tensor[float]
    Y: shape[N2, F]. Tensor[float]
    XX: ij,ij->i. 对F做内积, N1逐位运算. shape[N1]
    YY: ij,ij->i. shape[N2]
    return: shape[N1, N2]
    """
    # dist(x, y) = (x-y)^2 = sqrt(inner(x, x) - 2 * inner(x, y) + inner(y, y)). x, y: [F]
    if XX is None:
        XX = torch.einsum("ij,ij->i", X, X)
    if YY is None:
        YY = torch.einsum("ij,ij->i", Y, Y)
    # 减少分配空间导致的效率下降
    res = X @ Y.T
    res.mul_(-2).add_(XX[:, None]).add_(YY)
    res.clamp_min_(0.)  # 避免sqrt{负数}, 浮点误差
    return res if squared else res.sqrt_()


# if __name__ == "__main__":
#     from torchmetrics.functional.pairwise.euclidean import pairwise_euclidean_distance as _pairwise_euclidean_distance
#     x = torch.randn(10000, 1000)
#     x2 = torch.randn(10000, 1000)
#     y1 = libs_ml.test_time(lambda: pairwise_euclidean_distance(x, x2), 5)
#     y2 = libs_ml.test_time(lambda: _pairwise_euclidean_distance(x, x2), 5)
#     y3 = libs_ml.test_time(lambda: F.pairwise_distance(x, x2), 5)
#     print(torch.allclose(y1, y2))  # True
#     print(torch.allclose(y2.ravel()[::10000+1], y3, atol=1e-6))  # True
#     y4 = libs_ml.test_time(lambda: batched_euclidean_distance(x, x2), 5)
#     print(torch.allclose(y4, y3, atol=1e-6))  # True


def batched_mse(y_pred: Tensor, y_true: Tensor, reduction: Literal["mean", "sum"] = "mean") -> Tensor:
    """
    pred: [N, F]. float
    target: [N, F]. float
    return: [N]
    """
    F = y_pred.shape[1]
    res = batched_euclidean_distance(y_pred, y_true, squared=True)
    if reduction == "mean":
        res.div_(F)
    else:
        assert reduction == "sum"
    return res


def pairwise_mse(y_pred: Tensor, y_true: Tensor, reduction: Literal["mean", "sum"] = "mean") -> Tensor:
    """
    pred: [N, F]. float
    target: [N, F]. float
    return: [N, N]
    """
    F = y_pred.shape[1]
    res = pairwise_euclidean_distance(y_pred, y_true, squared=True)
    if reduction == "mean":
        res.div_(F)
    else:
        assert reduction == "sum"
    return res


# if __name__ == "__main__":
#     x = torch.randn((2000, 1000))
#     x2 = torch.randn((2000, 1000))
#     y = libs_ml.test_time(lambda: batched_mse(x, x2), number=10)
#     y2 = libs_ml.test_time(lambda: F.mse_loss(x, x2, reduction="none").mean(dim=1), number=10)
#     y3 = libs_ml.test_time(lambda: pairwise_mse(x, x2), number=10)
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, y3.diag()))


def pairwise_equal(X: Tensor, Y: Tensor) -> Tensor:
    """
    X: [N], y: [M]
    return: [N, M]
    """
    return X[:, None] == Y

# if __name__ == '__main__':
#     x = torch.arange(1000)
#     x2 = torch.arange(5, 1005)
#     y = libs_ml.test_time(lambda: pairwise_equal(x, x2), number=10)
#     print(y)

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

"""
1. 信息量 Info: [F]->[F]: -log(X)
2. 熵 Entropy: [F]->[F]: (X*Info(X)).sum()
3. 交叉熵 Cross_Entropy: [F],[F]->[F]: (T*Info(P)).sum()
4. 相对熵/KL散度: [F],[F]->[F]: (T*(Info(P)-INFO(T))).sum()=Cross_Entropy(T,P)-Entropy(T)
"""


def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    """KL(P||Q)=(P * log(P/Q)).sum(). P:数据分布(e.g. y_true). Q:P的先验后近似分布(e.g. 均匀分布)
    p: [N, F]
    q: [N, F]
    """
    N = p.shape[0]
    return ((p/q).log_().mul_(p)).sum().div_(N)


# if __name__ == "__main__":
#     from torchmetrics.functional import kl_divergence as _kl_divergence
#     X = torch.rand(1000, 100)
#     Y = torch.rand(1000, 100)
#     X = torch.softmax(X, dim=1)
#     Y = torch.softmax(Y, dim=1)
#     print(libs_ml.test_time(lambda: _kl_divergence(X, Y)))
#     print(libs_ml.test_time(lambda: kl_divergence(X, Y)))


def pearson_corrcoef(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """见_functional.py:corrcoef
    y_pred: [N] or [N,F]
    y_true: [N] or [N,F]
    return: [] or [F]    
    """
    N = y_pred.shape[0]
    p_mean = y_pred.mean(dim=0)
    t_mean = y_true.mean(dim=0)
    #
    p_std = y_pred.std(dim=0)  # N-1
    t_std = y_true.std(dim=0)
    res = (y_pred - p_mean).mul_(y_true - t_mean).sum(dim=0).div_(N - 1)
    res.div_(p_std).div_(t_std)
    return res


# if __name__ == "__main__":
#     from torchmetrics.functional import pearson_corrcoef as _pearson_corrcoef
#     target = torch.randn(10000)
#     preds = torch.randn(10000)
#     y = libs_ml.test_time(lambda: _pearson_corrcoef(preds, target))
#     y2 = libs_ml.test_time(lambda: pearson_corrcoef(preds, target))
#     y3 = libs_ml.test_time(lambda: torch.corrcoef(torch.stack([preds, target])))
#     print(torch.allclose(y, y2))
#     print(torch.allclose(y, y3[0, 1]))
#     target = torch.randn(10000, 1000)
#     preds = torch.randn(10000, 1000)
#     y = libs_ml.test_time(lambda: _pearson_corrcoef(preds, target))
#     y2 = libs_ml.test_time(lambda: pearson_corrcoef(preds, target))
#     print(torch.allclose(y, y2))

def calc_rank(x: Tensor) -> Tensor:
    """faster"""
    N = x.shape[0]
    x, idx = x.sort()
    rank = torch.empty_like(x)
    rank[idx] = torch.arange(1, N+1, dtype=x.dtype, device=x.device)
    _calc_rank_loop(x.numpy(), rank.numpy(), idx.numpy())
    return rank


# def calc_rank(x: Tensor) -> Tensor:
#     """
#     x: [N]
#     return: [N]
#     """
#     N = x.shape[0]
#     sorted_x, idx = x.sort()
#     rank = torch.empty_like(x)
#     rank[idx] = torch.arange(1, N + 1, dtype=x.dtype, device=x.device)
#     #
#     repeat_value: Tensor = sorted_x[:-1][sorted_x.diff() == 0]
#     repeat_value = repeat_value.unique_consecutive()
#     for r in repeat_value:
#         cond = torch.nonzero(x == r, as_tuple=True)
#         rank[cond] = rank[cond].mean()
#     return rank


# if __name__ == "__main__":
#     from torchmetrics.functional.regression.spearman import _rank_data as rank_data
#     # x = torch.randint(0, 100, (10000,)).float()
#     x = torch.tensor([5, 1, 4, 5.])
#     y = libs_ml.test_time(lambda: rank_data(x), 10)
#     y2 = libs_ml.test_time(lambda: calc_rank(x), 10)
#     print(torch.allclose(y, y2))


def spearman_corrcoef(y_pred: Tensor, y_true: Tensor) -> Tensor:
    """Ref: https://zh.m.wikipedia.org/zh-hans/%E6%96%AF%E7%9A%AE%E5%B0%94%E6%9B%BC%E7%AD%89%E7%BA%A7%E7%9B%B8%E5%85%B3%E7%B3%BB%E6%95%B0
    y_pred: [N] or [N,F]
    y_true: [N] or [N,F]
    return: [] or [F]
    # 对rank计算pearson_corrcoef(rank=1(from 1):最小值, 相同的数的rank使用它们的均值)
    # 性质: 
    1. 斯皮尔曼等级相关系数为1表明两个被比较的变量是单调相关的, 即使它们之间的相关关系可能并非线性的. 相较而言，其皮尔逊相关关系并不完美. 
    2. 当数据大致呈椭圆分布且没有明显的离群点时, 皮尔逊相关系数的值和斯皮尔曼相关系数的值接近.
    3. 对样本中的显著离群点, 斯皮尔曼相关系数比皮尔逊相关系数不敏感.
    4. 正的斯皮尔曼相关系数反映两个变量X和Y之间单调递增的趋势; 负的斯皮尔曼相关系数反映两个变量X和Y之间单调递减的趋势
    """
    if y_pred.ndim == 1:
        y_pred = calc_rank(y_pred)
        y_true = calc_rank(y_true)
    else:
        y_pred = torch.stack([calc_rank(yp) for yp in y_pred.unbind(dim=1)], dim=1)
        y_true = torch.stack([calc_rank(yt) for yt in y_true.unbind(dim=1)], dim=1)
    return pearson_corrcoef(y_pred, y_true)


# if __name__ == "__main__":
#     from torchmetrics.functional import spearman_corrcoef as _spearman_corrcoef
#     target = torch.randn(10000)
#     preds = torch.randn(10000)
#     y = libs_ml.test_time(lambda: _spearman_corrcoef(preds, target))
#     y2 = libs_ml.test_time(lambda: spearman_corrcoef(preds, target))
#     print(torch.allclose(y, y2))
#     target = torch.randn(10000, 1000)
#     preds = torch.randn(10000, 1000)
#     y = libs_ml.test_time(lambda: _spearman_corrcoef(preds, target))
#     y2 = libs_ml.test_time(lambda: spearman_corrcoef(preds, target))
#     print(torch.allclose(y, y2))
#     #
#     target = torch.randn(2000 * 1000)
#     preds = torch.randn(2000 * 1000)
#     y = libs_ml.test_time(lambda: _spearman_corrcoef(preds, target))
#     y2 = libs_ml.test_time(lambda: spearman_corrcoef(preds, target))
#     print(torch.allclose(y, y2))
