from heapq import nlargest, nsmallest, merge
from heapq import heapify, heappush, heappop, heappushpop, heapreplace
from bisect import bisect_left
from typing import Generic, List, TypeVar, Optional, Dict, Any, Tuple
import heapq

__all__ = [
    "siftdown", "siftup", "siftdown_max", "siftup_max",
    "heapify", "heappush", "heappop", "heappushpop", "heapreplace",
    "heapify_max", "heappush_max", "heappop_max", "heappushpop_max", "heapreplace_max",
    "PriorityQueue"
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


def _parent(i: int) -> int:
    """1,2->0"""
    return (i - 1) // 2


def _lc(i: int) -> int:
    """rc = lc + 1"""
    return (2 * i) + 1


class PriorityQueue(Generic[T]):
    def __init__(self, nums: Optional[List[T]] = None, max_heap: bool = False) -> None:
        if nums is None:
            nums = []

        if max_heap:
            self._heapify = heapify_max
            self._heappush = heappush_max
            self._heappop = heappop_max
        else:
            self._heapify = heapify
            self._heappush = heappush
            self._heappop = heappop
        self.heap = nums
        self._heapify(self.heap)

    def add(self, v: T) -> None:
        self._heappush(self.heap, v)

    def pop(self) -> T:
        return self._heappop(self.heap)

    def peek(self) -> T:
        return self.heap[0]

    def __len__(self) -> int:
        return len(self.heap)


if __name__ == "__main__":
    pq = PriorityQueue([1, 2, 3], max_heap=True)
    pq.add(10)
    pq.add(4)
    print(pq.peek())
    print(pq.pop())
    print(pq.heap)


class MutablePQ(Generic[T]):
    """小根堆实现"""

    def __init__(self) -> None:
        self.heap: List[Tuple[int, T]] = []  # 存储id
        self.id_to_idx: Dict[int, int] = {}  # id映射到index

    def _siftdown(self, i: int, lo: int) -> None:
        """见`heapq_.py`. 上滤
        -: 每次上滤, heap中含id, key. 先存储heap[i]的值.
            将i的父节点大于的, 赋值到空缺处, 并且在赋值时, 同时调整idx的值的变换.
        """
        heap = self.heap
        id_to_idx = self.id_to_idx
        #
        x = heap[i]  # id
        while True:
            p = _parent(i)
            px = heap[p]
            if p < lo or px[1] <= x[1]:
                break
            #
            heap[i] = px
            id_to_idx[px[0]] = i
            i = p
        heap[i] = x
        id_to_idx[x[0]] = i

    def _siftup(self, i: int, hi: Optional[int] = None) -> None:
        """见`heapq_.py`. 下滤"""
        heap = self.heap
        id_to_idx = self.id_to_idx
        #
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
            if rc <= hi and heap[c][1] < heap[rc][1]:
                c = rc
            #
            heap[i] = heap[c]
            id_to_idx[heap[c][0]] = i
            i = c
        heap[i] = x
        id_to_idx[x[0]] = i
        self._siftdown(i, lo)

    def _heappush(self, id_: int, v: T) -> None:
        n = len(self.heap)
        self.id_to_idx[id_] = n
        self.heap.append((id_, v))
        self._siftdown(n, 0)

    def add(self, id_: int, v: T):
        self._heappush(id_, v)

    def _heappop(self) -> Tuple[int, T]:
        heap = self.heap
        #
        res = heap.pop()
        if self.heap:
            res, heap[0] = heap[0], res
            self._siftup(0)
        del self.id_to_idx[res[0]]
        return res

    def pop(self) -> Tuple[int, T]:
        return self._heappop()

    def peek(self) -> Tuple[int, T]:
        return self.heap[0]

    def increase_key(self, k: int, v: T) -> None:
        pass


if __name__ == "__main__":
    mpq = MutablePQ()
    mpq.add(1, 5)
    mpq.add(2, 4)
    mpq.add(3, 4)
    mpq.add(4, 6)
    mpq.add(5, 1)
    print(mpq.id_to_idx)
    print(mpq.heap)
    print(mpq.pop())
    print(mpq.id_to_idx)
    print(mpq.heap)
