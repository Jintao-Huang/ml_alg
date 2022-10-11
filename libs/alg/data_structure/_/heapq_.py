"""小根堆的实现
Ref: 标准库: heapq; 做了一些小修改. 包括: siftup, siftdown的名称符合教材叫法.
"""

from typing import TypeVar, List, Optional


__all__ = []

T = TypeVar("T", float, int)


def _parent(i: int) -> int:
    """1,2->0"""
    return (i - 1) // 2


def _lc(i: int) -> int:
    """rc = lc + 1"""
    return (2 * i) + 1


def siftup(heap: List[T], i: int, lo: int = 0) -> None:
    """上滤. [lo..i]
        将heap[i]的元素进行上滤, 直到i < lo.
    note: 与python heapq中命名不同, heapq中为_siftdown
    -: 常发生在push时. 将i与父节点比较, 如果i更小, 则上滤. 直到p<lo.
    """
    x = heap[i]
    while True:
        p = _parent(i)
        if p < lo or heap[p] <= x:
            break
        #
        heap[i] = heap[p]
        i = p
    heap[i] = x


if __name__ == "__main__":
    from heapq import _siftdown
    heap = [1, 2, 3, 4, 5, 6, 0]
    _siftdown(heap, 0, 6)
    print(heap)
    heap = [1, 2, 3, 4, 5, 6, 0]
    siftup(heap, 6)
    print(heap)


def siftdown(heap: List[T], i: int, hi: Optional[int] = None) -> None:
    """下滤 (比python实现多了hi参数, 用于heapsort). [i..hi]
    -: 为什么在最后调用siftup. Ref: https://stackoverflow.com/questions/71632226/why-does-the-python-heapq-siftup-call-siftdown-at-the-end
        因为下滤常发生在pop时. 而i通常是应该在下面的节点. 该实现可以减少比较次数.
        (虽然我感觉差不多...)
    -: 使用左右子节点的最小值不断与i交换, 直到c>hi. 随后再进行上滤
    """
    if hi is None:
        hi = len(heap) - 1
    # 
    lo = i
    x = heap[i]
    while True:
        c = _lc(i)
        if c > hi:
            break
        #
        rc = c + 1
        if rc <= hi and heap[rc] < heap[c]:
            c = rc
        #
        heap[i] = heap[c]
        i = c
    heap[i] = x
    siftup(heap, i, lo)


def siftdown2(heap: List[T], i: int, hi: Optional[int] = None) -> None:
    """heapify使用可以加快速度"""
    if hi is None:
        hi = len(heap) - 1
    # 
    x = heap[i]
    while True:
        c = _lc(i)
        if c > hi:
            break
        #
        rc = c + 1
        if rc <= hi and heap[rc] < heap[c]:
            c = rc
        #
        if heap[c] >= x:
            break
        heap[i] = heap[c]
        i = c
    heap[i] = x


if __name__ == "__main__":
    from heapq import _siftup
    heap = [6, 0, 1, 2, 3, 4, 5]
    _siftup(heap, 0)
    print(heap)
    heap = [6, 0, 1, 2, 3, 4, 5]
    siftdown(heap, 0)
    print(heap)
    heap = [6, 0, 1, 2, 3, 4, 5]
    siftdown2(heap, 0)
    print(heap)


def heapify(heap: List[T]) -> None:
    for i in reversed(range(_parent(len(heap) - 1) + 1)):
        siftdown2(heap, i)  # faster than siftdown


def heapify2(heap: List[T]) -> None:
    for i in reversed(range(_parent(len(heap) - 1) + 1)):
        siftdown(heap, i)


if __name__ == "__main__":

    import mini_lightning as ml
    import numpy as np
    ml.seed_everything(42)
    l = np.random.randn(100000).tolist()
    ml.test_time(lambda: heapify(l), 10)
    #
    ml.seed_everything(42)
    l = np.random.randn(100000).tolist()
    ml.test_time(lambda: heapify2(l), 10)


def heap_sort(arr: List[T]) -> None:
    """从大到小排序. Os(1)
    -: 建最小堆. 然后将不断将最大元素放到后面(交换), 并下滤. 
    """
    heapify(arr)
    for i in reversed(range(1, len(arr))):
        arr[0], arr[i] = arr[i], arr[0]
        siftdown(arr, 0, i - 1)


def heap_sort2(arr: List[T]) -> None:
    """从大到小排序. Os(1)
    -: 建最小堆. 然后将不断将最大元素放到后面(交换), 并下滤. 
    """
    heapify(arr)
    for i in reversed(range(1, len(arr))):
        arr[0], arr[i] = arr[i], arr[0]
        siftdown2(arr, 0, i - 1)
"""使用heapq实现的版本见sort.py"""

if __name__ == "__main__":
    import mini_lightning as ml
    import numpy as np
    ml.seed_everything(42)
    l = np.random.randn(100000).tolist()
    ml.test_time(lambda: heap_sort(l), 1)
    #
    ml.seed_everything(42)
    l = np.random.randn(100000).tolist()
    ml.test_time(lambda: heap_sort2(l), 1)
