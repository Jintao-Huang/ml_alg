from libs import *
from libs.alg import *


class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = [[] for _ in range(numCourses)]
        for to, from_ in prerequisites:
            graph[from_].append(to)
        return topo_sort(graph)


if __name__ == "__main__":
    numCourses = 4
    prerequisites = [[1, 0], [2, 0], [3, 1], [3, 2]]
    print(Solution().findOrder(numCourses, prerequisites))
