

# ROUTE, PPL, BLEU, BPC

if __name__ == "__main__":
    from torchmetrics.text.rouge import ROUGEScore
    import torchmetrics.functional.text.bleu as bleu
    import torchmetrics.functional.text.rouge as rouge
    from pprint import pprint

from typing import List, Literal, Dict, Tuple, Union
from torch import Tensor
import torch
from collections import Counter

__all__ = []


def _create_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    n_tokens = len(tokens)
    res = Counter()
    for i in range(n_tokens - n + 1):
        key = tuple(tokens[i:i+n])
        res[key] += 1
    return res

# if __name__ == "__main__":
#     tokens = ["I", "am", "Huang", "Jintao"]
#     print(_create_ngrams(tokens, 1))
#     print(_create_ngrams(tokens, 2))
#     print(_create_ngrams(tokens, 4))
#     print(_create_ngrams(tokens, 5))


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


def _rouge_n_score(
    pred: List[str],
    target: List[str],
    n_gram: int
) -> Dict[str, Tensor]:
    """return: Dict的key: precision, recall, fmeasure(f1). """
    # pred, target里是token的List. 首先建立ngrams. 即: 连续的n个词为整体进行计数.
    # 得到pred和target的计数. 随后看pred, target中匹配的计数数量有多少. 该数量作为tp.
    # t1(t=1)为len(target), p1(p=1)为len(pred)
    pred_ngrams = _create_ngrams(pred, n_gram)
    target_ngrams = _create_ngrams(target, n_gram)
    tp = 0
    p1 = sum(pred_ngrams.values())
    t1 = len(target_ngrams.values())
    for pn in pred_ngrams.keys():
        if pn in target_ngrams:
            tp += min(pred_ngrams[pn], target_ngrams[pn])
    #
    tp = torch.tensor(tp, dtype=torch.float32)
    prec = _precision(tp, p1)
    recall = _recall(tp, t1)
    fmeasure = _fscore(prec, recall)
    return {
        "precision": prec, "recall": recall, "fmeasure": fmeasure
    }

# if __name__ == "__main__":
#     pred = "My name is John".split()
#     target = "Is My name John".split()
#     print(rouge._rouge_n_score(pred, target, 1))
#     print(_rouge_n_score(pred, target, 1))
#     print(rouge._rouge_n_score(pred, target, 2))
#     print(_rouge_n_score(pred, target, 2))
#     print(rouge._rouge_n_score(pred, target, 5))
#     print(_rouge_n_score(pred, target, 5))


def _lcs(pred_tokens: List[str], target_tokens: List[str], return_full_table: bool = False
         ) -> Union[int, List[List[int]]]:
    """Longest Common Subsequence"""
    # dp[0][0] = 0
    # 若pt[i]==tt[j], 则dp[i+1][j+1]=dp[i][j]+1. 否则dp[i+1][j+1]=max(dp[i+1][j], dp[i][j+1])
    p_len = len(pred_tokens)
    t_len = len(target_tokens)
    dp = [[0] * (t_len + 1) for _ in range(p_len + 1)]
    # dp[0][0] = 0
    for i in range(p_len):
        for j in range(t_len):
            ip, jp = i + 1, j + 1  # plus
            if pred_tokens[i] == target_tokens[j]:
                dp[ip][jp] = dp[i][j] + 1
            else:
                dp[ip][jp] = max(dp[i][jp], dp[ip][j])
    return dp[-1][-1]


# if __name__ == "__main__":
#     pred = "My name is John".split()
#     target = "Is My name John".split()
#     print(_lcs(pred, target))


def _rouge_l_score(
    pred: List[str],
    target: List[str]
) -> Dict[str, Tensor]:
    p1 = len(pred)
    t1 = len(target)
    tp = _lcs(pred, target, False)
    #
    tp = torch.tensor(tp, dtype=torch.float32)
    prec = _precision(tp, p1)
    recall = _recall(tp, t1)
    fmeasure = _fscore(prec, recall)
    return {
        "precision": prec, "recall": recall, "fmeasure": fmeasure
    }


# if __name__ == "__main__":
#     pred = "My name is John".split()
#     target = "Is My name John".split()
#     print(rouge._rouge_l_score(pred, target))
#     print(_rouge_l_score(pred, target))

###
# def rouge_score(
#     preds: List[str],
#     targets: List[List[str]],
#     accumulate: Literal["avg", "best"] = "best",
# ):
#     """accumulate: 'best': 取preds对应target中最好的. 'avg': 取preds对应target的平均"""
#     pass


# if __name__ == "__main__":
#     preds = "My name is John"
#     targets = ["Is your name John", "Is my name John"]

#     # pprint(_rouge_score_update([preds], [[targets]], [1, 2, "L", "Lsum"], "best"))
#     pprint(rouge.rouge_score([preds], [targets], "best"))

#     # rouge = ROUGEScore()
#     # pprint(rouge(preds, target))


# if __name__ == "__main__":
#     preds = "My name is John"
#     targets = ["Is your name John", "Is my name John"]

#     pprint(bleu.bleu_score([preds], [targets], n_gram=1))
#     import collections
#     collections.Counter()
