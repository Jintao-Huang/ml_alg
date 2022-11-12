# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import List, Tuple
from bisect import bisect_left
try:
    from .._utils._climit import INT32_INF
except ImportError:
    from libs.alg._utils._climit import INT32_INF
__all__ = [
    "LIS", "LIS2",
    "LCS", "LCS2",
    "edit_distance",
    "matrix_chain", "matrix_chain2"
]


def LIS(nums: List[int]) -> int:
    """使用非递增栈. 当然这个递减栈中只存栈顶元素(最小的元素)
    Test Ref: https://leetcode.cn/problems/longest-increasing-subsequence/
        https://leetcode.cn/problems/russian-doll-envelopes/
    """
    res = []  # 一定是递增的(二分查找)
    for x in nums:
        i = bisect_left(res, x)
        if i == len(res):
            res.append(x)
        else:
            res[i] = x
    return len(res)


def LIS2(nums: List[int]) -> int:
    """
    -: dp[i]: 以i结尾的LIS. 
        已知dp[0..i]确定dp[i+1]. 遍历j:0..i, 若nums[i]>nums[j]: dp[i+1]=max(dp[j]+1, dp[i+1]). 
        dp[i+1]初始化为1.
    Test Ref: https://leetcode.cn/problems/longest-increasing-subsequence/
    """
    n = len(nums)
    dp = [0] * n
    for i in range(n):
        dp[i] = 1
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)


if __name__ == "__main__":
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(LIS(nums))


if __name__ == "__main__":
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    print(LIS(nums))
    print(LIS2(nums))


def LCS(s1: str, s2: str) -> int:
    """
    -: dp[0][0]=0表示空字符串, dp[0][j]=0, dp[i][0]=0. dp[i][j]表示s1[0..i-1] s2[0..j-1]的LCS. 
        若s1[i-1]==s2[j-1]. 则dp[i][j]=dp[i-1][j-1] + 1. 否则=max(dp[i-1][j],do[i][j-1])
    Test Ref: https://leetcode.cn/problems/longest-common-subsequence/
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
                continue
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]


def _rebuild_LCS(s1: str, rb: List[List[int]]) -> str:
    res = []
    i, j = len(rb) - 1, len(rb[0]) - 1
    while i > 0 and j > 0:
        if rb[i][j] == 0:
            res.append(s1[i - 1])
            i -= 1
            j -= 1
        elif rb[i][j] == 1:
            i -= 1
        else:
            j -= 1
    return "".join(reversed(res))


def LCS2(s1: str, s2: str) -> Tuple[int, str]:
    """加了返回LCS
    Ref: 算法导论: 动态规划
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # 0表示同时前进. 1表示s1前进. 2表示s2前进.
    rb = [[0] * (m + 1) for _ in range(n + 1)]  # rebuild
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
                continue
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i-1][j-1] + 1
                rb[i][j] = 0
            elif dp[i - 1][j] > dp[i][j - 1]:
                dp[i][j] = dp[i - 1][j]
                rb[i][j] = 1
            else:
                dp[i][j] = dp[i][j - 1]
                rb[i][j] = 2

    return dp[n][m], _rebuild_LCS(s1, rb)


if __name__ == "__main__":
    text1 = "abcde"
    text2 = "ace"
    print(LCS(text1, text2))
    print(LCS2(text1, text2))


def edit_distance(s1: str, s2: str) -> int:
    """将s1 -> s2的编辑距离. (可以插入一个字符, 删除一个字符, 替换一个字符)
    -: 空字符到空字符的编辑距离为0. dp[0][0]=0. 
        dp[i][j]表示s1[0..i-1]->s2[0..j-1]的编辑距离
        若s[i-1]==s[j-1]. 则dp[i][j]=dp[i-1][j-1]
        否则: dp[i][j]=min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]) + 1
    Test Ref: https://leetcode.cn/problems/edit-distance/
    """
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0:
                dp[0][j] = j
                continue
            if j == 0:
                dp[i][0] = i
                continue
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
    return dp[n][m]


if __name__ == "__main__":
    word1 = "horse"
    word2 = "ros"
    print(edit_distance("horse", "ros"))


def matrix_chain(nums: List[int]) -> int:
    """返回矩阵乘的运算数量
    Ref: 算法导论: 动态规划
    -: 10 20 30表示: 10*20,20*30的矩阵乘法. 等于10*20*30
        dp[i][j]表示i..j的最小乘法数量.
        dp[i][i]表示i..i, =0, dp[i][i+1]=0
        dp[i][j]可以使用k遍历i..j. =max(dp[i][k]*dp[k][j]+nums[i]*nums[k]*nums[j])
    Test Ref: https://blog.csdn.net/luoshixian099/article/details/46344175
        用于测试
    """
    n = len(nums)
    dp = [[0] * n for _ in range(n)]  # 初始化dp[i][i], dp[i][i+1]
    #
    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            dp[i][j] = INT32_INF
            for k in range(i + 1, j):
                dp[i][j] = min(dp[i][j], dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j])
    return dp[0][n - 1]


def _rebuild_matmul(nums: List[int], rb: List[List[int]], i: int, j: int, res: List[str]) -> None:
    if i == j - 1:
        res.append(f"A{i}")
        return

    k = rb[i][j]
    res.append("(")
    _rebuild_matmul(nums, rb, i, k, res)
    _rebuild_matmul(nums, rb, k, j, res)
    res.append(")")


def matrix_chain2(nums: List[int]) -> Tuple[int, str]:
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    rb = [[-1] * n for _ in range(n)]  # rebuild
    #
    for l in range(2, n):
        for i in range(n - l):
            j = i + l
            dp[i][j] = INT32_INF
            for k in range(i + 1, j):
                v = dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j]
                if v < dp[i][j]:
                    dp[i][j] = v
                    rb[i][j] = k
    # rebuild
    res = []
    _rebuild_matmul(nums, rb, 0, n-1, res)

    return dp[0][n - 1], "".join(res)


if __name__ == "__main__":
    mc = [10, 20, 30]
    print(matrix_chain(mc))
    print(matrix_chain2(mc))
    mc = [30, 35, 15, 5, 10, 20, 25]
    print(matrix_chain(mc))
    print(matrix_chain2(mc))
