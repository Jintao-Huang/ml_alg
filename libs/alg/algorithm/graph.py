try:
    from ..data_structure import PriorityQueue
except ImportError:
    from libs.alg import PriorityQueue
from typing import List, Tuple, Union, Optional
from collections import namedtuple


WEdge = namedtuple("WEdge", ["to", "val"])
def dijkstra(graph: List[List[WEdge]], s: int, es: Optional[List[int]] = None) -> List[int]:
    """使用不可变的优先级队列实现. 
    -: 遍历所有的节点, 若某节点到s的距离最短, 则更新res, 并加入pq.
        pq使用到s的距离作为key. 若s已经被拓展, 则再遇到s则跳过. (pq中重复的)
    -: 数据结构: res, visited
    es: None表示遍历所有节点
    return: 返回所有节点到s的距离.
    """
    n = len(graph)
    es_s = set() if es is None else set(es)
    res = [-1] * n  # -1表示不
    visited = [False] * n  # 拓展的节点.
    pq = PriorityQueue([(0, s)])
    # 
    while len(pq) > 0:
        dist, from_ = pq.pop()
        if visited[from_]:  # 忽略不好的路径
            continue
        assert dist == res[from_]
        visited[from_] = True
        # 
        for to, d2 in graph[from_]:
            dist2 = dist + d2
            if dist2 >= res[to]:
                continue
            pq.add((dist, d2))
            res[to] = dist2





