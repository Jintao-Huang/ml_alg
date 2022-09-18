__all__ = []
from heapq import heapify, heappop, heappush, heappushpop, heapreplace
from heapq import _siftdown as __siftdown, _siftup as __siftup


def _parent(ci: int) -> int:
    """
    思路: 1,2 -> 0. p=(c-1) // 2
    """
    return (ci - 1) >> 1


def _lc(pi: int) -> int:
    """
    思路: 0->1; 1->3. lc=2p + 1
    """
    return pi >> 1 + 1


def _rc(pi: int) -> int:
    """
    思路: 0->2; 1->4
    """
    return (pi + 1) >> 1


def siftdown(heap, startpos, pos):
    pass


def siftup():
    pass


