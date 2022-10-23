

from typing import NamedTuple, TypeVar, List, Optional, Callable, Iterable
import math


__all__ = [
    "Point", "euclidean_distance", "manhattan_distance",
    "accumulate", "prefix_sum"
]

Point = NamedTuple("Point", x=int, y=int)


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
    p1 = Point(x=1, y=2)
    p2 = Point(x=4, y=6)
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
