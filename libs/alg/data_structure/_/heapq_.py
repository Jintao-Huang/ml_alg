

from typing import TypeVar, List, Any


__all__ = []

T = TypeVar("T")


def siftup(heap: List[Any], i: int, lo: int) -> None:
    """上滤
        将heap[i]的元素进行上滤, 直到i < lo.
        堆的边界: [lo..hi]
    note: 与python heapq中命名不同, heapq中为_siftdown
    -: 
    """


def siftdown(heap: List[Any], i: int, hi: int) -> None:
    """下滤
    -: 为什么在最后调用siftup. Ref: https://stackoverflow.com/questions/71632226/why-does-the-python-heapq-siftup-call-siftdown-at-the-end
    """


def heapify(heap: List[Any]) -> None:
    pass


def heappop(heap: List[T]) -> T:
    pass


def heappush(heap: List[T], x: T) -> None:
    pass


def heappushpop(heap: List[T], x: T) -> T:
    pass


def heapreplace(heap: List[T], x: T) -> T:
    pass
