try:
    from .._data_structure import PriorityQueue, MutablePQ, UFSet
except ImportError:  # for debug
    pass
from typing import List, Tuple, NamedTuple, Dict, Deque
from collections import deque, defaultdict

__all__ = ["WEdge", "WEdge2",
           "dijkstra", "dijkstra2", "dijkstra3",  "bfs",
           "kruskal", "prim", "prim2", "Dinic", "dinic", "hungarian"]

WEdge = NamedTuple("WEdge", to=int, val=int)  # val: e.g. 距离等
WEdge2 = NamedTuple("WEdge2", from_=int, to=int, val=int)


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
        复杂度: O((N+M)LogM), N为节点数量, M为边数量. (因为堆的最大长度为M)
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
        复杂度: O((N+M)LogN), N为节点数量, M为边数量. (可变的优先队列, 堆的最大长度为N)
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


def kruskal(n: int, edges: List[WEdge2]) -> int:
    """返回最小树的距离. 适用于稀疏图(无向图; edges中只需要一条边即可). 
    -: 传入边的list, 排序. 然后使用uf_set, 每次取最短的边, 直到uf_set的size=1
    复杂度: Ot(ELogE), E为len(edges)
    Test Ref: https://leetcode.cn/problems/min-cost-to-connect-all-points/
    """
    edges.sort(key=lambda e: e.val)
    ufset = UFSet(n)
    res = 0
    # 直到边都遍历, 或者uf_set的size为1
    for from_, to, val in edges:
        if ufset.size() == 1:
            break
        #
        if ufset.union(from_, to):
            res += val
    assert ufset.size() == 1  # 否则就是森林了.
    return res


def prim(graph: List[Dict[int, int]]) -> int:
    """适用于稠密图(无向图; 但graph需要有两条边; 使用邻接表2实现). 
        类似于dijkstra算法. 以某个定点为起点, 然后往外拓展.
    -: 以s为起点, 遍历周围的顶点, 更新pq. 
        然后从pq中选出最近的点, (若已在cost[v]==0, 表示已访问, 则忽略), 将cost[v]赋为0. 
        在遍历周围点时, 若发现visited, 则认为已经被访问, 忽略. 若长度>=cost[v]则忽略. 
    复杂度: Ot((N+M)LogM)
    Test Ref: https://leetcode.cn/problems/min-cost-to-connect-all-points/
    """
    s = 0
    n = len(graph)
    cost = [-1] * n
    cnt = 0
    res = 0
    pq = PriorityQueue[Tuple[int, int]]([(0, s)])
    #
    while len(pq) > 0 and cnt < n:
        dist, from_ = pq.pop()  # 拓展from_
        if cost[from_] == 0:
            continue
        cost[from_] = 0
        cnt += 1
        res += dist
        #
        for to, d2 in graph[from_].items():
            if cost[to] == 0:
                continue
            if cost[to] == -1 or d2 < cost[to]:
                pq.add((d2, to))

    assert cnt == n
    return res


def prim2(graph: List[Dict[int, int]], inf=int(1e9)) -> int:
    """适用于稠密图(无向图; 但graph需要有两条边; 使用邻接表2实现). 
        使用MutablePQ实现, 从而去除cost变量的使用.
    复杂度: Ot((N+M)LogN)
    Test Ref: https://leetcode.cn/problems/min-cost-to-connect-all-points/
    """
    s = 0
    n = len(graph)
    res = 0
    pq = MutablePQ[int]()
    pq.heap = [(i, inf) for i in range(n)]
    pq.id_to_idx = {i: i for i in range(n)}
    #
    pq.modify_priority(s, 0)
    #
    while len(pq) > 0:
        from_, dist = pq.pop()  # 拓展from_
        res += dist
        #
        for to, d2 in graph[from_].items():
            if to not in pq:
                continue
            if d2 < pq[to]:
                pq.modify_priority(to, d2)
    assert res < inf
    return res


"""最大流算法总结:
1. naive: 循环. 每次循环随便找一条简单路径s->t, 然后减掉这条路径的瓶颈流量. 更新residual graph. 
    循环直到找不到简单路径.
    特点: 只能保证找到阻塞流, 但不能保证是最大流.
2. Ford-Fulkerson: 
    主要思想: 通过引入反向路径, 后来可以撤销前面的路径.
        在更新RG的同时, 更新反向路径. e.g. 流量是3, 则增加反向路径3. 表示可以撤销3份流量.
        最后再去掉反向边即可得到RG. 
    规律: 正向流量(>=0) + 反向流量(>=0) = abs(原图的正向流量)
    特点: 保证是最大流, 但复杂度很高. 
    复杂度: O(fm). m为边数, f为最大流的大小.
3. Edmonds-Karp: 
    主要思想: 找简单路径时, 使用无权图的最短路. 
    复杂度: Ot(m^2n), m为边数, n为定点数.
4. Dinic: 
    主要思想: bfs中的level保证找的是最短路. dfs中多路增广+当前弧优化. 确保一次迭代不仅仅找一条路, 而是多条路的流量(阻塞流).
    直到bfs中找不到s->t的路径, 则退出训练. 找到的为最大流. 
    复杂度: Ot(mn^2)

最小割: 最小割=最大流.
"""


class Dinic:
    """可以解决有向有权图的最大流/最小割问题(权重代表流量); 以及无权二部图的匹配问题. (支持多重边)
        无权二部图的匹配问题, 可以建立两个虚拟的节点0(连接组0),1(连接组1). 然后求0->1的流大小. (组0, 组1间有连接)
        最大独立集=n-最大匹配/最小点覆盖
            最小点覆盖: 覆盖所有的边最小的点集
            最大独立集: 最大的点之间无边的点集
    Ref: https://www.bilibili.com/video/BV1K64y1C7Do/
    Test Ref: 
        1. 二部图: 
            https://leetcode.cn/problems/maximum-students-taking-exam/
            https://leetcode.cn/problems/broken-board-dominoes/
    """

    def __init__(self, n: int) -> None:
        # 这样设计的优势: 取反向边方便.
        self.edges: List[WEdge] = []  # 取反向边: i -> i^1. 所以不需要存from_
        # rg: residual graph
        self.rg: List[List[int]] = [[] for _ in range(n)]  # 里面存的是边的索引

    def add_edge(self, edge: WEdge2) -> None:
        from_, to, capacity = edge
        n = len(self.edges)
        self.rg[from_].append(n)
        self.rg[to].append(n + 1)

        self.edges.append(WEdge(to=to, val=capacity))
        self.edges.append(WEdge(to=from_, val=0))  # 反向边

    def _bfs(self, s: int, t: int) -> List[int]:
        """
        return: level. 若level[t]==-1, 则表示未找到s->t的路径
        """
        n = len(self.rg)
        level = [-1] * n
        visited = [False] * n
        visited[s] = True
        dq = deque([s])
        dist = 0  # 距离
        while len(dq) > 0:
            dq_len = len(dq)
            for _ in range(dq_len):
                v = dq.popleft()
                level[v] = dist
                if v == t:
                    break
                for idx in self.rg[v]:
                    e = self.edges[idx]
                    to, capacity = e
                    if capacity == 0:
                        continue
                    if visited[to]:
                        continue
                    visited[to] = True
                    dq.append(to)
            dist += 1
        return level

    def _dfs(self, v: int, t: int, flow: int, level: List[int], cur: List[int]) -> int:
        """
        flow: 表示前面的流. 
        cur: 用于当前弧优化. 即每个顶点的每条边, 一次dfs只会遍历一次. 
            这在二部图匹配中无作用, 但在最大流中有用. 
        return: 后面流的大小, 即消耗的流.
        """
        if v == t:
            return flow
        #
        flow_o = flow
        es = self.rg[v]
        for i in range(cur[v], len(es)):
            cur[v] += 1
            idx = es[i]
            e = self.edges[idx]
            if (level[e.to] - level[v] < 1):  # 即为回边
                continue
            if e.val == 0:
                continue
            #
            f = self._dfs(e.to, t, min(e.val, flow), level, cur)  # 后面的流
            #
            if f == 0:
                continue
            # f > 0: 建立回边
            e2 = self.edges[idx ^ 1]  # 异或
            new_e = WEdge(to=e.to, val=e.val-f)
            new_e2 = WEdge(to=e2.to, val=e2.val+f)
            self.edges[idx] = new_e
            self.edges[idx ^ 1] = new_e2
            flow -= f
            if flow == 0:
                break
        return flow_o - flow

    def run(self, s: int, t: int, inf=int(1e9)) -> int:
        """
        -: 不断通过循环: bfs, dfs进行flow的获取. 
            bfs: 得到level graph. 规定flow只能从低level流向高level.
            dfs: 获取当前迭代的flow. 并对边进行调整. 
            直到bfs从s到达不了t.

        """
        res = 0
        n = len(self.rg)
        while True:
            level = self._bfs(s, t)
            if level[t] == -1:
                break
            cur = [0] * n
            res += self._dfs(s, t, inf, level, cur)
        return res


def dinic(graph: List[Dict[int, int]], s: int, t: int) -> int:
    """
    graph: 有向图
    """
    n = len(graph)
    _dinic = Dinic(n)
    for from_ in range(len(graph)):
        es = graph[from_]
        for to, capacity in es.items():
            _dinic.add_edge(WEdge2(from_, to, capacity))
    return _dinic.run(s, t)


def _find(graph: List[List[int]], s: int, visited: List[bool], matching: List[int]) -> bool:
    """
    return: 是否成功找到
    """
    if visited[s]:
       return False 
    visited[s] = True
    for v in graph[s]:
        # v未匹配, 或matching[v]可以挪位置
        if matching[v] == -1 or _find(graph, matching[v], visited, matching):
            matching[s] = v
            matching[v] = s
            return True
    return False


def hungarian(graph: List[List[int]]) -> int:
    """匈牙利算法(这也是无权二部图匹配算法)
        思想: 挪位置法. 回溯法(只不过每次find时, 已经挪过位置的不能再挪了) 
    graph: 存一条边, g1->g2的有向边
    -: 不断遍历每一个节点, 对于每一个节点, 采用回溯法. 
        若可以匹配, 则返回. 若不能匹配, 则令已经匹配的点依次挪位置
    复杂度: O(nm)
    """
    res = 0
    n = len(graph)
    matching = [-1] * n  # 匹配情况
    for s in range(n):
        # 类似于当前弧优化. 避免回溯, 使得每条边, 一次find只会遍历一次. (且充当visited, 已访问点)
        visited = [False] * n
        if _find(graph, s, visited, matching):
            res += 1
    return res
