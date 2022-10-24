
from libs import *
from libs.alg import *


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        return quick_select(nums, len(nums) - k)


class Solution2:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        res = heapq.nlargest(k, nums)
        return res[-1]


if __name__ == "__main__":
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    print(Solution().findKthLargest(nums, k))
    print(Solution2().findKthLargest(nums, k))
