import bisect
from typing import List, TypeVar, Optional

__all__ = []

T = TypeVar("T")


def bisect_left(arr: List[T], x: T, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最小的>=x的值
    Test Ref: https://leetcode.cn/problems/search-insert-position/
    """
    if hi is None:
        hi = len(arr)
    #
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bisect_right(arr: List[T], x: T, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最小的>x的值
    """
    if hi is None:
        hi = len(arr)
    #
    lo -= 1
    hi -= 1
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] > x:
            hi = mid
        else:
            lo = mid + 1
    return lo


def bisect_left2(arr: List[T], x: T, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最大<x的值 + 1
    Test Ref: https://leetcode.cn/problems/search-insert-position/
    """
    if hi is None:
        hi = len(arr)
    #
    lo -= 1
    hi -= 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if arr[mid] < x:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


def bisect_right2(arr: List[T], x: T, lo: int = 0, hi: Optional[int] = None) -> int:
    """[lo..hi).
    -: 找到res(i), 使得res是最大的<=x的值 + 1
    """
    if hi is None:
        hi = len(arr)
    #
    lo -= 1
    hi -= 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if arr[mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo + 1


if __name__ == "__main__":
    from bisect import bisect_left as _bisect_left, bisect_right as _bisect_right
    x = [0, 1, 1, 2]
    print(bisect_left(x, 1))
    print(bisect_left2(x, 1))
    print(_bisect_left(x, 1))
    print(bisect_right(x, 1))
    print(bisect_right2(x, 1))
    print(_bisect_right(x, 1))
