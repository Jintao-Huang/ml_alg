try:
    from ..data_structure import PriorityQueue, MutablePQ
except ImportError:  # for debug
    pass
from typing import List, Tuple, NamedTuple, Dict, Deque
from collections import deque

__all__ = ["WEdge", "dijkstra", "dijkstra2", "dijkstra3"]

WEdge = NamedTuple("WEdge", to=int, val=int)  # val: e.g. 距离等


def dijkstra(graph: List[List[WEdge]], s: int) -> List[int]:
    """使用不可变的优先级队列实现. (使用邻接表实现), dist需要>=0
    -: 遍历所有的节点, 若某节点到s的距离最短, 则更新res, 并加入pq.
        pq使用v到s的距离作为key. 若v已经被拓展(visited), 则再遇到v时则跳过. (pq中重复的)
    return: 返回所有节点到s的距离. inf表示未找到
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    """
    n = len(graph)
    res = [-1] * n
    res[s] = 0
    visited = [False] * n  # 拓展的节点(pq中可能含有多个相同的顶点). 当然不使用visited数组区别不大, 可能略微增加计算量.
    pq = PriorityQueue([(0, s)])
    #
    while len(pq) > 0:
        dist, from_ = pq.pop()  # 拓展from_
        if visited[from_]:  # 忽略不好的路径: e.g. 先入队列, 但后出队列的情况. MutablePQ可以避免这一问题.
            continue
        assert dist == res[from_]
        visited[from_] = True
        #
        for to, d2 in graph[from_]:
            dist2 = dist + d2
            if res[to] != -1 and dist2 >= res[to]:
                continue
            pq.add((dist2, to))
            res[to] = dist2

    return res


def dijkstra2(graph: List[Dict[int, int]], s: int) -> List[int]:
    """使用不可变的优先级队列实现. (使用邻接表2实现), dist需要>=0
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    """
    n = len(graph)
    res = [-1] * n
    res[s] = 0
    visited = [False] * n
    pq = PriorityQueue[Tuple[int, int]]([(0, s)])
    #
    while len(pq) > 0:
        dist, from_ = pq.pop()
        if visited[from_]:
            continue
        assert dist == res[from_]
        visited[from_] = True
        #
        for to, d2 in graph[from_].items():
            dist2 = dist + d2
            if res[to] != -1 and dist2 >= res[to]:
                continue
            pq.add((dist2, to))
            res[to] = dist2

    return res


def dijkstra3(graph: List[Dict[int, int]], s: int) -> List[int]:
    """使用可变的优先级队列实现. (使用邻接表2实现), dist需要>=0
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    note: 在python中这么做会降低速度, 但在cpp中会加快速度. (原因是我们重写了heapq中的c函数, 使用python实现)
    """
    n = len(graph)
    res = [-1] * n
    res[s] = 0
    pq = MutablePQ[int]()
    pq.add(s, 0)
    #
    while len(pq) > 0:
        from_, dist = pq.pop()  # 拓展from_
        #
        for to, d2 in graph[from_].items():
            dist2 = dist + d2
            if res[to] != -1 and dist2 >= res[to]:
                continue
            if to in pq:
                pq.modify_priority(to, dist2)
            else:
                pq.add(to, dist2)
            res[to] = dist2

    return res


def bfs(graph: List[List[int]], s: int) -> List[int]:
    """无权图的最短路. 邻接表存储: 无向图需要存两条边
    -: 使用visited数组, deque. 使用每次拓展一个位置的方式进行拓展.
        在入队列时修改res, visited. 即队列中不存在已被访问过的节点.
    """
    n = len(graph)
    res = [-1] * n
    res[s] = 0
    dq = Deque[int]([s])
    visited = [False] * n
    visited[s] = True
    dist = 1
    while len(dq) > 0:
        dq_len = len(dq)
        for _ in range(dq_len):
            v = dq.popleft()
            for v2 in graph[v]:
                if visited[v2]:
                    continue
                visited[v2] = True
                res[v2] = dist
                dq.append(v2)
        dist += 1
    return res


if __name__ == "__main__":
    graph = [[1, 3, 4], [0, 2], [1], [0, 5], [0, 5], [3, 4]]
    print(bfs(graph, 0))
    print(bfs(graph, 3))
