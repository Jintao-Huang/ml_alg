from typing import Callable

__all__ = ["bs", "bs2"]


def bs(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    """返回满足cond的下界索引. 范围: [lo..hi]
    Test Ref: https://leetcode.cn/problems/koko-eating-bananas/
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bs2(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    """返回满足cond的上界索引. 范围: [lo..hi]
    Test Ref: https://leetcode.cn/problems/sum-of-scores-of-built-strings/
    """
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo