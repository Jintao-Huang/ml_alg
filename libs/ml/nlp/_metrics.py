# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from typing import List, Literal, Dict, Tuple, Union
from torch import Tensor
import torch
from collections import Counter
import re
import math

"""ROUTE, BLEU, PPL, BPW, BPC
ROUTE: 翻译和摘要的metric.
    1,2,L的precision, recall, fmeasure都在0-1之间. 越高越好.
BLEU: 翻译的metric, 在0-1之间. 越高越好.
PPL: LM的metric, 越低越好.
BPW: LM的metric, 越低越好.
BPC: LM的metric, 越低越好.
"""


__all__ = []

if __name__ == "__main__":
    from torchmetrics.text.rouge import ROUGEScore
    import torchmetrics.functional.text.bleu as bleu
    import torchmetrics.functional.text.rouge as rouge
    from pprint import pprint


def _create_ngrams(tokens: List[str], n: int) -> Counter[Tuple[str, ...]]:
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
    pred: List[str],  # 一个样例
    target: List[str],  # pred对应target中的一个. 而不考虑"best" or "avg"
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
    #
    tp = sum((pred_ngrams & target_ngrams).values())
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


def _preprocessing(
    sentence: str
) -> List[str]:
    """切割sentence为token list"""
    res = re.split(r"[^a-z0-9]+", sentence.lower())
    return [token for token in res if len(token) > 0]

# if __name__ == "__main__":
#     print(_preprocessing(" 123 aAdf-=-=asdf a  "))

# if __name__ == "__main__":
#     preds = ["My name is John", "cat cat cat"]
#     targets = [["Is your name John", "Is my name John"], ["cat is me"]]
#     pprint(rouge.rouge_score(preds, targets, "best"))


def _bleu_score(
    pred: List[str],
    target: List[List[str]],
    n_gram: int
) -> Tensor:
    """
    pred: 一个样例; target: 对应pred的一组target
    """
    # 首先计算pred对应的ngrams(从1开始到n_gram), 以及target对应的ngrams.
    #   target的grams是其所有t的 |. 含义: 可以随意匹配target中的ngrams.
    # tp为pred_ngrams与target_ngrams的重叠. 并计算每个n_gram对应的tp和pi. 计算precision_n(对应n_gram)
    # 计算pred_len, 以及target_len的最小值, 并计算brevity_penalty.
    # BP = max(1, e^{1-tl/pl}). 即对于过短的pred进行惩罚. (pl < tl有惩罚)
    # 公式: BLEU: BP * exp(mean(logPn))
    pred_ngrams = Counter()
    for i in range(n_gram):
        pred_ngrams += _create_ngrams(pred, i + 1)  # |=, +=都可
    target_ngrams = Counter()
    for t in target:
        tmp_counter = Counter()
        for i in range(n_gram):
            tmp_counter += _create_ngrams(t, i + 1)  # |=, +=都可
        target_ngrams |= tmp_counter
    #
    tp_ngrams = pred_ngrams & target_ngrams
    tps = torch.zeros(n_gram)
    pi = torch.zeros(n_gram)
    n_pred = len(pred)
    n_target = min(len(t) for t in target)
    for tpn, v in tp_ngrams.items():
        tps[len(tpn) - 1] += v
    for pn, v in pred_ngrams.items():
        pi[len(pn) - 1] += v
    #
    prec_n = _precision(tps, pi)  # [n_gram]
    bp = max(1, math.exp(1-n_target / n_pred))  # 修正过短的pred.
    # 为什么使用log: 让其更关注低分. (同交叉熵损失)
    res = bp * (prec_n.log_().mean().exp_())
    return res
# if __name__ == "__main__":
#     preds = "my name is John"
#     targets = ["is your name John", "is my name John hahaha"]
#     pprint(bleu.bleu_score([preds], [targets], n_gram=2))
#     preds = preds.split()
#     targets = [t.split() for t in targets]
#     pprint(_bleu_score(preds, targets, 2))


# 若是per-character预测的网络, 则PPL=e^{平均词长*bpc}. 这里bpc=loss
#   ref: https://arxiv.org/abs/1308.0850
# 若是per-word预测的网络, 则PPL=e^bpw. 这里bpw=loss. loss=mean(-log(ps)).
def perplexity(
    bpw: Tensor
) -> Tensor:
    """ppl
    bpw: []. bpw=mean(-log(p(x_i|x_<i))). 
        注意mean是对每个pos进行均值, 而不是d. 
    """
    # ref: https://huggingface.co/docs/transformers/perplexity
    # 公式: PPL(X)=exp(mean(-log(p(x_i|x_<i)))), 其中X=(x_0,...x_t), 共t个token.
    # 适用于自回归模型(LM)
    return bpw.exp()

# if __name__ == "__main__":
#     # bpw
#     loss = -torch.log(torch.tensor(0.6))
#     loss2 = -torch.log(torch.tensor(0.05))
#     print(loss, perplexity(loss))
#     print(loss2, perplexity(loss2))
#     # bpc
#     cpw = 5.6
#     print(perplexity(loss * cpw))
