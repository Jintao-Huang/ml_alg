import bisect
from typing import List, Optional
try:
    from .._binary_search import bs, bs2
except ImportError:  # for debug
    from libs.alg._algorithm._binary_search import bs, bs2

__all__ = []


def bisect_left(nums: List[int], x: int, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最小的>=x的值
    Test Ref: https://leetcode.cn/problems/search-insert-position/
    """
    if hi is None:
        hi = len(nums)
    return bs(lo, hi, lambda i: nums[i] >= x)


def bisect_right(nums: List[int], x: int, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最小的>x的值
    """
    if hi is None:
        hi = len(nums)
    #
    return bs(lo, hi, lambda i: nums[i] > x)


def bisect_left2(nums: List[int], x: int, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最大<x的值 + 1
    Test Ref: https://leetcode.cn/problems/search-insert-position/
    """
    if hi is None:
        hi = len(nums)
    #
    return bs2(lo - 1, hi - 1, lambda i: nums[i] < x) + 1


def bisect_right2(nums: List[int], x: int, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最大的<=x的值 + 1
    """
    if hi is None:
        hi = len(nums)
    #
    return bs2(lo - 1, hi - 1, lambda i: nums[i] <= x) + 1


if __name__ == "__main__":
    from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right
    x = [0, 1, 1, 2]
    print(bisect_left(x, 1))
    print(bisect_left2(x, 1))
    print(_bisect_left(x, 1))
    print(bisect_right(x, 1))
    print(bisect_right2(x, 1))
    print(_bisect_right(x, 1))
