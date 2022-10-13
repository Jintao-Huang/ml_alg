

from typing import NamedTuple, TypeVar
import math


__all__ = ["Point", "euclidean_distance", "manhattan_distance"]

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
