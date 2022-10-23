from libs import *
from libs.alg import *

class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        s = sum(nums)
        if s % 2 == 1:
            return False
        #
        C = s // 2
        init_value = -INT32_INF
        res = k01_full_max(nums, C, init_value=init_value)
        return False if res == init_value else True

if __name__ == "__main__":
    nums = [1,5,11,5]
    print(Solution().canPartition(nums))
    nums = [1,2,3,5]
    print(Solution().canPartition(nums))