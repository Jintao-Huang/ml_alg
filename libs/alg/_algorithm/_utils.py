# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import NamedTuple, TypeVar, List, Optional, Callable, Iterable
import math

import random
try:
    from ._._sort import partition
except ImportError:
    from libs.alg._algorithm._._sort import partition
__all__ = [
    "Point", "euclidean_distance", "manhattan_distance",
    "accumulate", "prefix_sum",
    "quick_select",
]

Point = NamedTuple("Point", x1=int, x2=int)


def euclidean_distance(p1: Point, p2: Point, square: bool = False) -> float:
    d1, d2 = (p1.x - p2.x), (p1.y - p2.y)
    res = d1 * d1 + d2 * d2
    if not square:
        res = math.sqrt(res)
    return res


def manhattan_distance(p1: Point, p2: Point) -> int:
    d1, d2 = (p1.x - p2.x), (p1.y - p2.y)
    return abs(d1) + abs(d2)


if __name__ == "__main__":
    p1 = Point(1, 2)
    p2 = Point(4, 6)
    print(euclidean_distance(p1, p2))
    print(manhattan_distance(p1, p2))

T = TypeVar("T")


def accumulate(
    nums: Iterable[T],
    accumulate_func: Optional[Callable[[T, int], int]] = None,
    res: Optional[List[int]] = None,
    start: int = 0
) -> List[int]:
    """
    Test Ref: _data_structure/_string_hasher.py
    """
    if accumulate_func is None:
        accumulate_func: Callable[[T, int], int] = lambda x, y: x + y
    if res is None:
        res = []
    #
    for y in nums:
        x = start if len(res) == 0 else res[-1]
        z = accumulate_func(x, y)
        res.append(z)
    return res


def prefix_sum(nums: List[int], include_zero: bool = True) -> List[int]:
    if include_zero:
        return accumulate(nums, None, [0])
    else:
        return accumulate(nums, None, None, 0)


if __name__ == "__main__":
    nums = [1, 2, 3, 4]
    print(prefix_sum(nums))
    print(prefix_sum(nums, False))


def quick_select(nums: List[int], idx: int) -> int:
    """顺序统计量
    Test Ref: https://leetcode.cn/problems/kth-largest-element-in-an-array/
    """
    lo, hi = 0, len(nums) - 1
    assert lo <= idx <= hi
    while True:
        r = random.randint(lo, hi)
        nums[r], nums[lo] = nums[lo], nums[r]
        pivot = partition(nums, lo, hi)
        if pivot == idx:
            return nums[idx]
        elif idx < pivot:
            hi = pivot - 1
        else:
            lo = pivot + 1


if __name__ == "__main__":
    nums = [1, 3, 5, 7, 9, 8, 6, 4, 2]
    print(quick_select(nums, 5))
