
from libs import *
from libs.alg import *


class Solution:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        -: 找所有点的最小生成树. 图为完全无向图
        """
        edges = []
        n = len(points)
        for i, (x, y) in enumerate(points):
            p1 = Point(x, y)
            for j in range(i + 1, n):
                x2, y2 = points[j]
                p2 = Point(x2, y2)
                dist = manhattan_distance(p1, p2)
                edges.append(WEdge2(i, j, dist))

        return kruskal(n, edges)


class Solution2:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        -: 找所有点的最小生成树. 图为完全无向图
        """
        n = len(points)
        graph = [{} for _ in range(n)]
        for i, (x, y) in enumerate(points):
            p1 = Point(x, y)
            for j in range(i + 1, n):
                x2, y2 = points[j]
                p2 = Point(x2, y2)
                dist = manhattan_distance(p1, p2)
                graph[i][j] = dist
                graph[j][i] = dist

        return prim(graph)

class Solution3:
    def minCostConnectPoints(self, points: List[List[int]]) -> int:
        """
        -: 找所有点的最小生成树. 图为完全无向图
        """
        n = len(points)
        graph = [{} for _ in range(n)]
        for i, (x, y) in enumerate(points):
            p1 = Point(x, y)
            for j in range(i + 1, n):
                x2, y2 = points[j]
                p2 = Point(x2, y2)
                dist = manhattan_distance(p1, p2)
                graph[i][j] = dist
                graph[j][i] = dist

        return prim2(graph)

if __name__ == "__main__":
    points = [[11, -6], [9, -19], [16, -13], [4, -9], [20, 4], [20, 7], [-9, 18], [10, -15], [-15, 3], [6, 6]]
    print(Solution().minCostConnectPoints(points))
    print(Solution2().minCostConnectPoints(points))
    print(Solution3().minCostConnectPoints(points))
