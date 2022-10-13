from libs import *
from libs.alg import *


class Solution:
    def domino(self, n: int, m: int, broken: List[List[int]]) -> int:
        """任务是建立二部图. 然后使用dinic算法
        -: 使用i+j是奇数还是偶数构建二部图. 然后计算匹配的个数.
        -: 图: 奇数流向偶数. 
        """
        b_s = {tuple(b) for b in broken}
        dinic = Dinic(n * m + 2)  # 2分别为奇偶. 0为偶, 1为奇. 令s=0, t=1
        for i in range(n):
            for j in range(m):
                if (i, j) in b_s:
                    continue
                idx = i * m + j + 2  # 定点的索引
                t = (i + j) % 2   # 类型
                if t == 0:
                    dinic.add_edge(WEdge2(0, idx, 1))
                else:
                    dinic.add_edge(WEdge2(idx, 1, 1))
                # 右边, 下边的连线.
                if j < m-1 and (i, j + 1) not in b_s:
                    if t == 0:
                        e = WEdge2(idx, idx + 1, 1)
                    else:
                        e = WEdge2(idx + 1, idx, 1)
                    dinic.add_edge(e)
                if i < n - 1 and (i + 1, j) not in b_s:
                    if t == 0:
                        e = WEdge2(idx, idx + m, 1)
                    else:
                        e = WEdge2(idx + m, idx, 1)
                    dinic.add_edge(e)
        return dinic.run(0, 1)


class Solution2:
    def domino(self, n: int, m: int, broken: List[List[int]]) -> int:
        """匈牙利算法"""
        b_s = {tuple(b) for b in broken}
        graph = [[] for _ in range(n * m)]
        for i in range(n):
            for j in range(m):
                if (i, j) in b_s:
                    continue
                idx = i * m + j  # 定点的索引
                t = (i + j) % 2   # 类型
                # 右边, 下边的连线.
                if j < m-1 and (i, j + 1) not in b_s:
                    if t == 0:
                        graph[idx].append(idx + 1)
                    else:
                        graph[idx + 1].append(idx)
                if i < n - 1 and (i + 1, j) not in b_s:
                    if t == 0:
                        graph[idx].append(idx + m)
                    else:
                        graph[idx + m].append(idx)
        return hungarian(graph)


if __name__ == "__main__":
    n = 2
    m = 3
    broken = [[1, 0], [1, 1]]
    print(Solution().domino(n, m, broken))
    print(Solution2().domino(n, m, broken))
