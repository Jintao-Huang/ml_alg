from libs import *
from libs.alg import *


class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        res = prev_k_max(nums, k)
        return [nums[res[i]] for i in range(len(res)) if i + 1 >= k]


if __name__ == "__main__":
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    print(Solution().maxSlidingWindow(nums, k))
