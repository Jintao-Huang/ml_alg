# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Generic, List, TypeVar, Optional, Dict, Any, Tuple
try:
    from ._heapq import *
    from ._ import _parent, _lc
except ImportError:
    from libs.alg._data_structure._heapq import *
    from libs.alg._data_structure._ import _parent, _lc
__all__ = ["PriorityQueue", "MutablePQ"]


T = TypeVar("T")


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


# if __name__ == "__main__":
#     pq = PriorityQueue([1, 2, 3], max_heap=True)
#     pq.add(10)
#     pq.add(4)
#     print(pq.peek())
#     print(pq.pop())
#     print(pq.heap)


K, V = TypeVar("K"), TypeVar("V")


class MutablePQ(Generic[K, V]):
    """小根堆实现: v越小, 优先级越高
    -: 通过唯一标识符id, 优先级v. 得到的优先级队列.
        比PQ增加的功能: 可以通过id, 对v的优先级进行调整
    Test Ref: 
        Graph dijkstra: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
        Graph prim: https://leetcode.cn/problems/min-cost-to-connect-all-points/
    """

    def __init__(self) -> None:
        self.heap: List[Tuple[K, V]] = []  # 存储id
        self.id_to_idx: Dict[K, int] = {}  # id映射到index
        # 当然也可以通过id, 获取index后, 从heap中获得优先级v.

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
            if rc <= hi and heap[rc][1] < heap[c][1]:
                c = rc
            #
            heap[i] = heap[c]
            id_to_idx[heap[c][0]] = i
            i = c
        heap[i] = x
        id_to_idx[x[0]] = i
        self._siftdown(i, lo)

    def _heappush(self, id_: K, v: V) -> None:
        n = len(self.heap)
        self.id_to_idx[id_] = n
        self.heap.append((id_, v))
        self._siftdown(n, 0)

    def add(self, id: K, v: V):
        self._heappush(id, v)

    def _heappop(self) -> Tuple[K, V]:
        heap = self.heap
        #
        res = heap.pop()
        if self.heap:
            res, heap[0] = heap[0], res
            self._siftup(0)
        del self.id_to_idx[res[0]]
        return res

    def pop(self) -> Tuple[K, V]:
        return self._heappop()

    def peek(self) -> Tuple[K, V]:
        return self.heap[0]

    def modify_priority(self, id: K, v: V) -> None:
        idx = self.id_to_idx[id]
        _, v_o = self.heap[idx]
        self.heap[idx] = id, v
        if v_o < v:  # 降低优先级, 下滤
            self._siftup(idx)
        else:
            self._siftdown(idx, 0)

    def __getitem__(self, id: K) -> V:
        return self.heap[self.id_to_idx[id]][1]

    def __contains__(self, id: K) -> bool:
        return id in self.id_to_idx

    def __len__(self) -> int:
        return len(self.heap)


if __name__ == "__main__":
    mpq = MutablePQ()
    mpq.add(1, 5)
    print(mpq.id_to_idx)
    print(mpq.heap)
    mpq.add(2, 4)
    mpq.add(3, 4)
    mpq.add(4, 6)
    mpq.add(5, 1)
    print(mpq.id_to_idx)
    print(mpq.heap)
    print(mpq.pop())
    print(mpq.id_to_idx)
    print(mpq.heap)
    mpq.modify_priority(1, 0)
    print(mpq.id_to_idx)
    print(mpq.heap)
    mpq.modify_priority(1, 5)
    print(mpq.id_to_idx)
    print(mpq.heap)
