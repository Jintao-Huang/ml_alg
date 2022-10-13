from libs import *
from libs.alg import *


class Solution:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        """使用邻接表
        -: 获取scr1, scr2, dest到所有节点的距离.
            依次遍历每一个节点, 使得该节点到src1, src2, dest的距离最短.
        """
        graph: List[List[WEdge]] = [[] for _ in range(n)]
        graph_r: List[List[WEdge]] = [[] for _ in range(n)]
        for e in edges:
            from_, to, val = e
            graph[from_].append(WEdge(to=to, val=val))
            graph_r[to].append(WEdge(to=from_, val=val))
        INF = int(1e12)
        res3 = dijkstra(graph_r, dest)
        if res3[src1] == -1 or res3[src2] == -1:
            return -1
        # 一定能找到解
        res1 = dijkstra(graph, src1)
        res2 = dijkstra(graph, src2)
        res = INF
        for i in range(n):
            a = res1[i]
            b = res2[i]
            c = res3[i]
            if a == -1 or b == -1 or c == -1:
                continue
            res = min(a + b + c, res)
        return res


def bfs(graph: List[Dict[int, int]], s: int) -> List[int]:
    """(使用邻接表实现)
    Test Ref: https://leetcode.cn/problems/minimum-weighted-subgraph-with-the-required-paths/
    """
    n = len(graph)
    res = [-1] * n
    res[s] = 0
    dq: Deque[Tuple[int, int]] = deque([(0, s)])
    #
    while len(dq) > 0:
        dist, from_ = dq.popleft()
        #
        for to, d2 in graph[from_].items():
            dist2 = dist + d2
            if res[to] != -1 and dist2 >= res[to]:
                continue
            dq.append((dist2, to))
            res[to] = dist2

    return res


dijkstra2 = bfs


class Solution2:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        """使用邻接表矩阵"""
        graph: List[Dict[int, int]] = [{} for i in range(n)]
        graph_r: List[Dict[int, int]] = [{} for i in range(n)]
        INF = int(1e12)
        for e in edges:
            from_, to, val = e
            graph[from_][to] = min(graph[from_].get(to, INF), val)
            graph_r[to][from_] = min(graph_r[to].get(from_, INF), val)
        res3 = dijkstra2(graph_r, dest)
        if res3[src1] == -1 or res3[src2] == -1:
            return -1
        res1 = dijkstra2(graph, src1)
        res2 = dijkstra2(graph, src2)
        res = INF
        for i in range(n):
            a = res1[i]
            b = res2[i]
            c = res3[i]
            if a == -1 or b == -1 or c == -1:
                continue
            res = min(a + b + c, res)
        return res


class Solution3:
    def minimumWeight(self, n: int, edges: List[List[int]], src1: int, src2: int, dest: int) -> int:
        """使用邻接表矩阵"""
        graph: List[Dict[int, int]] = [{} for i in range(n)]
        graph_r: List[Dict[int, int]] = [{} for i in range(n)]
        INF = int(1e12)
        for e in edges:
            from_, to, val = e
            graph[from_][to] = min(graph[from_].get(to, INF), val)
            graph_r[to][from_] = min(graph_r[to].get(from_, INF), val)
        res3 = dijkstra3(graph_r, dest)
        if res3[src1] == -1 or res3[src2] == -1:
            return -1
        res1 = dijkstra3(graph, src1)
        res2 = dijkstra3(graph, src2)
        res = INF
        for i in range(n):
            a = res1[i]
            b = res2[i]
            c = res3[i]
            if a == -1 or b == -1 or c == -1:
                continue
            res = min(a + b + c, res)
        return res


if __name__ == "__main__":
    # n = 6
    # edges = [[0, 2, 2], [0, 5, 6], [1, 0, 3], [1, 4, 5], [2, 1, 1], [2, 3, 3], [2, 3, 4], [3, 4, 2], [4, 5, 1]]
    # src1 = 0
    # src2 = 1
    # dest = 5
    # print(Solution2().minimumWeight(n, edges, src1, src2, dest))
    # #
    # n = 3
    # edges = [[0, 1, 1], [2, 1, 1]]
    # src1 = 0
    # src2 = 1
    # dest = 2
    # print(Solution2().minimumWeight(n, edges, src1, src2, dest))
    n = 5
    edges = [[0, 1, 2], [1, 3, 2], [2, 3, 3], [3, 4, 2], [3, 4, 3]]
    src1 = 0
    src2 = 2
    dest = 4
    print(Solution().minimumWeight(n, edges, src1, src2, dest))
    print(Solution2().minimumWeight(n, edges, src1, src2, dest))
    print(Solution3().minimumWeight(n, edges, src1, src2, dest))
