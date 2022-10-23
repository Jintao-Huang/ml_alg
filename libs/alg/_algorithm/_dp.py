from typing import List
from bisect import bisect_left

__all__ = ["LIS", "LIS2", "LCS"]


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


if __name__ == "__main__":
    text1 = "abcde"
    text2 = "ace"
    print(LCS(text1, text2))


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
