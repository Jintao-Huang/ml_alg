
from libs import *
from libs.alg import *


class Solution:
    def trap(self, height: List[int]) -> int:
        """按列求
        -: 对于每个坐标i, 查找其下一个>=该坐标j, 上一个>该坐标k. 然后水当前行的量: h*w
            其中h=min(height[j], height[k]]) - height[i], w=j-k-1
        """
        nge, pgt = next_ge_prev_gt(height)  # ge, gt都是一样的, 因为h为0.
        n = len(height)
        res = 0
        for i in range(n - 1):
            j = nge[i]
            k = pgt[i]
            if j != -1 and k != -1:
                w = j - k - 1
                h = min(height[j], height[k]) - height[i]
                res += w * h
        return res


if __name__ == "__main__":
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(Solution().trap(height))
    #
    height = [4, 2, 0, 3, 2, 5]
    print(Solution().trap(height))
