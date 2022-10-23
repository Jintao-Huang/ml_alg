from libs import *
from libs.alg import *

class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        n, m = len(matrix), len(matrix[0])
        hs = [0] * m
        res= 0
        for i in range(n):
            for j in range(m):
                if matrix[i][j] == "1":
                    hs[j] += 1
                else:
                    hs[j] = 0
            res = max(res, largest_rect(hs))
        return res
    
if __name__ == "__main__":
    matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
    print(Solution().maximalRectangle(matrix))



