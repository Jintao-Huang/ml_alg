import heapq
from heapq import nlargest, nsmallest, merge
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
from typing import TypeVar, List
__all__ = [
    "siftdown", "siftup", "siftdown_max", "siftup_max",
    "heapify", "heappush", "heappop", "heappushpop", "heapreplace",
    "heapify_max", "heappush_max", "heappop_max", "heappushpop_max", "heapreplace_max",
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


if __name__ == "__main__":
    """test"""
    import mini_lightning as ml
    import numpy as np

    def heap_sort(arr: List[T], dst: List[T]) -> None:
        heapify(arr)
        for _ in range(len(arr)):
            dst.append(heappop(arr))
    ml.seed_everything(42)
    x: List[float] = np.random.rand(100000).tolist()
    dst = []
    ml.test_time(lambda: heap_sort(x, dst), 10)
    #

    def heap_sort2(arr: List[T], dst: List[T]) -> None:
        heapify_max(arr)
        for _ in range(len(arr)):
            dst.append(heappop_max(arr))
    ml.seed_everything(42)
    x: List[float] = np.random.rand(100000).tolist()
    dst = []
    ml.test_time(lambda: heap_sort2(x, dst), 10)
