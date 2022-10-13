from libs import *
from libs.alg import *


class Solution:
    def maxStudents(self, seats: List[List[str]]) -> int:
        """以列建立二部图"""
        n, m = len(seats), len(seats[0])
        dinic = Dinic(n * m + 2)  # 0表示偶, 1表示奇. 0 -> 1
        cnt = 0
        for i in range(n):
            for j in range(m):
                if seats[i][j] == "#":
                    continue
                cnt += 1
                t = j % 2
                idx = i * m + j + 2
                if t == 0:
                    dinic.add_edge(WEdge2(0, idx, 1))
                else:
                    dinic.add_edge(WEdge2(idx, 1, 1))
                #
                if i > 0:
                    if j > 0 and seats[i - 1][j - 1] == ".":
                        if t == 0:
                            e = WEdge2(idx, idx - 1-m, 1)
                        else:
                            e = WEdge2(idx - 1-m, idx, 1)
                        dinic.add_edge(e)
                    if j < m - 1 and seats[i - 1][j + 1] == ".":
                        if t == 0:
                            e = WEdge2(idx, idx + 1-m, 1)
                        else:
                            e = WEdge2(idx + 1-m, idx, 1)
                        dinic.add_edge(e)
                # 向左的: 向右的已经包括了.
                if j < m - 1 and seats[i][j + 1] == ".":
                    if t == 0:
                        e = WEdge2(idx, idx + 1, 1)
                    else:
                        e = WEdge2(idx + 1, idx, 1)
                    dinic.add_edge(e)
        return cnt - dinic.run(0, 1)


class Solution2:
    def maxStudents(self, seats: List[List[str]]) -> int:
        """以列建立二部图"""
        n, m = len(seats), len(seats[0])
        cnt = 0
        graph = [{} for i in range(n * m + 2)]
        for i in range(n):
            for j in range(m):
                if seats[i][j] == "#":
                    continue
                cnt += 1
                t = j % 2
                idx = i * m + j + 2
                if t == 0:
                    graph[0][idx] = 1
                else:
                    graph[idx][1] = 1
                #
                if i > 0:
                    if j > 0 and seats[i - 1][j - 1] == ".":
                        if t == 0:
                            graph[idx][idx - 1-m] = 1
                        else:
                            graph[idx - 1-m][idx] = 1
                    if j < m - 1 and seats[i - 1][j + 1] == ".":
                        if t == 0:
                            graph[idx][idx + 1-m] = 1
                        else:
                            graph[idx+1-m][idx] = 1
                # 向左的: 向右的已经包括了.
                if j < m - 1 and seats[i][j + 1] == ".":
                    if t == 0:
                        graph[idx][idx + 1] = 1
                    else:
                        graph[idx+1][idx] = 1
        return cnt - dinic(graph, 0, 1)


class Solution3:
    def maxStudents(self, seats: List[List[str]]) -> int:
        n, m = len(seats), len(seats[0])
        cnt = 0
        graph = [[] for i in range(n * m)]
        for i in range(n):
            for j in range(m):
                if seats[i][j] == "#":
                    continue
                cnt += 1
                t = j % 2
                idx = i * m + j
                #
                if i > 0:
                    if j > 0 and seats[i - 1][j - 1] == ".":
                        if t == 0:
                            graph[idx].append(idx - 1-m)
                        else:
                            graph[idx - 1-m].append(idx)
                    if j < m - 1 and seats[i - 1][j + 1] == ".":
                        if t == 0:
                            graph[idx].append(idx + 1-m)
                        else:
                            graph[idx+1-m].append(idx)
                # 向左的: 向右的已经包括了.
                if j < m - 1 and seats[i][j + 1] == ".":
                    if t == 0:
                        graph[idx].append(idx + 1)
                    else:
                        graph[idx+1].append(idx)
        return cnt - hungarian(graph)


if __name__ == "__main__":
    seats = [["#", ".", "#", "#", ".", "#"],
             [".", "#", "#", "#", "#", "."],
             ["#", ".", "#", "#", ".", "#"]]

    print(Solution().maxStudents(seats))
    print(Solution2().maxStudents(seats))
    print(Solution3().maxStudents(seats))
