# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import heapq
from heapq import nlargest, nsmallest, merge
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
from typing import TypeVar, List
__all__ = [
    "siftdown", "siftup", "siftdown_max", "siftup_max",
    "heapify", "heappush", "heappop", "heappushpop", "heapreplace",
    "heapify_max", "heappush_max", "heappop_max", "heappushpop_max", "heapreplace_max",
    "heap_sort"
]

siftdown = heapq._siftdown
siftup = heapq._siftup
siftdown_max = heapq._siftdown_max
siftup_max = heapq._siftup_max
#
try:
    heapify_max = heapq._heapify_max
except AttributeError:
    def heapify_max(x):
        n = len(x)
        for i in reversed(range(n//2)):
            siftup_max(x, i)
#
T = TypeVar("T")
try:
    heapreplace_max = heapq._heapreplace_max
except AttributeError:
    def heapreplace_max(heap: List[T], x: T) -> T:
        x, heap[0] = heap[0], x
        siftup_max(heap, 0)
        return x
try:
    heappop_max = heapq._heappop_max
except AttributeError:
    def heappop_max(heap: List[T]) -> T:
        res = heap.pop()
        if not heap:
            return res
        #
        res, heap[0] = heap[0], res
        siftup_max(heap, 0)
        return res


def heappush_max(heap: List[T], x: T) -> None:
    heap.append(x)
    siftdown_max(heap, 0, len(heap) - 1)


def heappushpop_max(heap: List[T], x: T) -> T:
    if not heap or heap[0] <= x:
        return x
    #
    x, heap[0] = heap[0], x
    siftup_max(heap, 0)
    return x


def heap_sort(nums: List[T], dst: List[T], reversed: bool = False) -> None:
    if reversed:
        _heapify = heapify_max
        _heappop = heappop_max
    else:  # 从小到大
        _heapify = heapify
        _heappop = heappop
    _heapify(nums)
    for _ in range(len(nums)):
        dst.append(_heappop(nums))


if __name__ == "__main__":
    """test"""
    import mini_lightning as ml
    import numpy as np

    ml.seed_everything(42)
    x: List[float] = np.random.rand(100000).tolist()
    dst = []
    ml.test_time(lambda: heap_sort(x, dst), 10)
    print(dst[:100])
    #
    ml.seed_everything(42)
    x: List[float] = np.random.rand(100000).tolist()
    dst = []
    ml.test_time(lambda: heap_sort(x, dst, True), 10)
    print(dst[:100])
