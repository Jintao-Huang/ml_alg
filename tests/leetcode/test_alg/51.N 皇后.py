from libs import *
from libs.alg import *


class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        res_int: List[List[int]] = n_queens(n)
        res = []
        s = "." * n
        for ri in res_int:
            r = []
            for i in ri:
                r.append(s[:i] + "Q" + s[i+1:])
            res.append(r)
        return res


if __name__ == "__main__":
    print(Solution().solveNQueens(4))
