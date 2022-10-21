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
    dp = [1] * n  # 初始化
    for i in range(n):
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


def LCS():
    """
    Test Ref: https://leetcode.cn/problems/longest-common-subsequence/
    """
    pass
