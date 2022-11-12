# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import List, Callable, Iterable, Optional
try:
    from .._utils._climit import INT32_INF
except ImportError:  # for debug
    from libs.alg._utils._climit import INT32_INF

__all__ = [
    "k01", "k01_full_max", "k01_full_min",
    "kC", "kC_full_max", "kC_full_min",
    "kC2", "kC_full_max2", "kC_full_min2",
    "kC_full_cnt"
]

"""
C: 背包容量
W: 每个背包的重量
V: 每个背包的价值
朴素的k01目标: 使得在W满足C的条件下, 使得V最大
"""


def _1(W: List[int], C: int, V: Optional[List[int]],
       iter_range: Callable[[int], Iterable], max_min_func: Callable[[int, int], int],
       C_full: bool, init_value: int) -> int:
    dp = [init_value] * (C+1)
    dp[0] = 0
    n = len(W)
    for i in range(n):
        w = W[i]
        v = V[i] if V is not None else 1
        for j in iter_range(C + 1):
            if j - w >= 0 and (not C_full or dp[j - w] != init_value):
                dp[j] = max_min_func(dp[j], dp[j - w] + v)
    return dp[C]


def _2(W: List[int], C: int, V: Optional[List[int]],
       max_min_func: Callable[[int, int], int],
       C_full: bool, init_value: int) -> int:
    """可以调换for的顺序. 只适用于完全背包"""
    dp = [init_value] * (C+1)
    dp[0] = 0
    n = len(W)
    for i in range(C + 1):
        for j in range(n):
            w = W[j]
            v = V[j] if V is not None else 1
            if i - w >= 0 and (not C_full or dp[i - w] != init_value):
                dp[i] = max_min_func(dp[i], dp[i - w] + v)
    return dp[C]


def k01(W: List[int], C: int, V: Optional[List[int]] = None) -> int:
    """01: 每个物品只能取1次; 使得在W满足<=C的条件下, 使得V最大"""
    return _1(W, C,  V, lambda n: reversed(range(n)), max, False, 0)


def k01_full_max(W: List[int], C: int, V: Optional[List[int]] = None, init_value=-INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最大
    Test Ref: https://leetcode.cn/problems/partition-equal-subset-sum/solution/
    """
    return _1(W, C, V, lambda n: reversed(range(n)), max, True, init_value)


def k01_full_min(W: List[int], C: int, V: Optional[List[int]] = None, init_value=INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最小"""
    return _1(W, C, V, lambda n: reversed(range(n)), min, True, init_value)


def kC(W: List[int], C: int, V: Optional[List[int]] = None) -> int:
    """完全背包: 每个物品只能取无限次, 使得在W满足<=C的条件下, 使得V最大"""
    return _1(W, C, V, lambda n: range(n), max, False, 0)


def kC2(W: List[int], C: int, V: Optional[List[int]] = None) -> int:
    return _2(W, C, V, max, False, 0)


def kC_full_max(W: List[int], C: int, V: Optional[List[int]] = None, init_value: int = -INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最大"""
    return _1(W, C, V, lambda n: range(n), max, True, init_value)


def kC_full_max2(W: List[int], C: int, V: Optional[List[int]] = None, init_value: int = -INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最大"""
    return _2(W, C, V, max, True, init_value)


def kC_full_min(W: List[int], C: int, V: Optional[List[int]] = None, init_value: int = INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最小
    Test Ref: https://leetcode.cn/problems/coin-change/
    """
    return _1(W, C, V, lambda n: range(n), min, True, init_value)


def kC_full_min2(W: List[int], C: int, V: Optional[List[int]] = None, init_value: int = INT32_INF) -> int:
    """使得在W满足==C的条件下, 使得V最大
    Test Ref: https://leetcode.cn/problems/coin-change/
    """
    return _2(W, C, V, min, True, init_value)


if __name__ == "__main__":
    W = [1, 2, 3, 4]
    V = [-1, -2, -4, -6]
    C = 5
    print(k01(W, C, V))
    print(k01_full_max(W, C, V))
    print(k01_full_min(W, C, V))
    #
    print(kC(W, C, V))
    print(kC2(W, C, V))
    print(kC_full_max(W, C, V))
    print(kC_full_max2(W, C, V))
    print(kC_full_min(W, C, V))
    print(kC_full_min2(W, C, V))
    print()
    W = [1, 2, 3, 4]
    V = [1, 2, 4, 6]
    C = 11
    print(k01(W, C, V))
    print(k01_full_max(W, C, V))
    print(k01_full_min(W, C, V))
    #
    print(kC(W, C, V))
    print(kC2(W, C, V))
    print(kC_full_max(W, C, V))
    print(kC_full_max2(W, C, V))
    print(kC_full_min(W, C, V))
    print(kC_full_min2(W, C, V))


def kC_full_cnt(W: List[int], C: int) -> int:
    """
    Test Ref: https://leetcode.cn/problems/coin-change-ii/
    """
    dp = [0] * (C+1)
    dp[0] = 1
    n = len(W)
    for i in range(n):
        w = W[i]
        for j in range(C + 1):
            if j - w >= 0:
                dp[j] += dp[j - w]
    return dp[C]


if __name__ == "__main__":
    amount = 5
    coins = [1, 2, 5]
    print(kC_full_cnt(coins, amount))
