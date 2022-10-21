from libs import *
from libs.alg import *


class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        prev_s = prefix_sum(nums)
        res = next_ge_k_len(prev_s, k)
        return res

if __name__ == "__main__":
    nums = [2,-1,2]
    k = 3
    print(Solution().shortestSubarray(nums, k))