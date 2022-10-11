from ..data_structure import PriorityQueue, MutablePQ
from typing import List, Tuple, Union, Optional, Any, NamedTuple, Dict, Deque
from collections import deque

__all__ = ["WEdge", "dijkstra", "dijkstra2", "dijkstra3"]

WEdge = NamedTuple("WEdge", to=int, val=int)  # val: e.g. 距离等


def dijkstra(graph: List[List[WEdge]], s: int, inf=int(1e9)) -> List[int]:
    """使用不可变的优先级队列实现. (使用邻接表实现)
    -: 遍历所有的节点, 若某节点到s的距离最短, 则更新res, 并加入pq.
        pq使用v到s的距离作为key. 若v已经被拓展(visited), 则再遇到v时则跳过. (pq中重复的)
    return: 返回所有节点到s的距离. inf表示未找到
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    """
    n = len(graph)
    res = [inf] * n
    res[s] = 0
    visited = [False] * n  # 拓展的节点(pq中可能含有多个相同的顶点)
    pq = PriorityQueue([(0, s)])
    #
    while len(pq) > 0:
        dist, from_ = pq.pop()  # 拓展from_
        if visited[from_]:  # 忽略不好的路径: e.g. 先入队列, 但后出队列的情况
            continue
        assert dist == res[from_]
        visited[from_] = True
        #
        for to, d2 in graph[from_]:
            dist2 = dist + d2
            if dist2 >= res[to]:
                continue
            pq.add((dist2, to))
            res[to] = dist2

    return res


def dijkstra2(graph: Dict[int, Dict[int, int]], n: int, s: int, inf=int(1e9)) -> List[int]:
    """使用不可变的优先级队列实现. (使用邻接表2实现)
    -: 遍历所有的节点, 若某节点到s的距离最短, 则更新res, 并加入pq.
        pq使用v到s的距离作为key. 若v已经被拓展(visited), 则再遇到v时则跳过. (pq中重复的)
    return: 返回所有节点到s的距离. inf表示未找到
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    """
    res = [inf] * n
    res[s] = 0
    visited = [False] * n  # 拓展的节点(pq中可能含有多个相同的顶点)
    pq = PriorityQueue[Tuple[int, int]]([(0, s)])
    #
    while len(pq) > 0:
        dist, from_ = pq.pop()  # 拓展from_
        if visited[from_]:  # 忽略不好的路径: e.g. 先入队列, 但后出队列的情况. MutablePQ可以避免这一问题.
            continue
        assert dist == res[from_]
        visited[from_] = True
        #
        for to, d2 in graph[from_].items():
            dist2 = dist + d2
            if dist2 >= res[to]:
                continue
            pq.add((dist2, to))
            res[to] = dist2

    return res


def dijkstra3(graph: Dict[int, Dict[int, int]], n: int, s: int, inf=int(1e9)) -> List[int]:
    """使用可变的优先级队列实现. (使用邻接表2实现)
    -: 遍历所有的节点, 若某节点到s的距离最短, 则更新res, 并加入pq.
        pq使用v到s的距离作为key. 若v已经被拓展(visited), 则再遇到v时则跳过. (pq中重复的)
    return: 返回所有节点到s的距离. inf表示未找到
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    note: 在python中这么做会降低速度, 但在cpp中会加快速度. (原因是我们重写了heapq中的c函数, 使用python实现)
    """
    res = [inf] * n
    res[s] = 0
    pq = MutablePQ[int]()
    pq.add(s, 0)
    #
    while len(pq) > 0:
        from_, dist = pq.pop()  # 拓展from_
        #
        for to, d2 in graph[from_].items():
            dist2 = dist + d2
            if dist2 >= res[to]:
                continue
            if to in pq:
                pq.modify_priority(to, dist2)
            else:
                pq.add(to, dist2)
            res[to] = dist2

    return res
