"""test UFSet"""

from libs import *
from libs.alg import *


class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        -: 遍历所有的点, 如果自己和邻居是OO, 则合并. 
            更n*m+1, 如果是和最后一个节点相连, 则为不相连的.
            需要过遍历两遍.
        """
        n, m = len(board), len(board[0])
        ufset = UFSet(n * m + 1)
        o = m * n
        for i in range(n):
            for j in range(m):
                if board[i][j] == "X":
                    continue
                #
                p0 = j + i * m
                ds = [(-1, 0), (0, -1)]
                if i == n - 1:
                    ds.append((1, 0))
                if j == m - 1:
                    ds.append((0, 1))
                # 
                for d in ds:
                    i2, j2 = d[0] + i, d[1] + j
                    #
                    if i2 < 0 or j2 < 0 or d[0] > 0 or d[1] > 0:
                        p1 = o
                    else:
                        if board[i2][j2] == "X":
                            continue
                        p1 = j2 + i2 * m
                    ufset.union(p0, p1)
                #

        #

        ro = ufset.find_root(o)
        for i in range(n):
            for j in range(m):
                p0 = j + i * m
                if board[i][j] == 'O' and ufset.find_root(p0) != ro:
                    board[i][j] = 'X'


if __name__ == "__main__":
    board = [["X", "X", "X", "X"], ["X", "O", "O", "X"], ["X", "X", "O", "X"], ["X", "O", "X", "X"]]
    Solution().solve(board)
    print(board)
