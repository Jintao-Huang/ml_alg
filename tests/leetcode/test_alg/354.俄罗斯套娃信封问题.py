from libs import *
from libs.alg import *


class Solution:
    def maxEnvelopes(self, envelopes: List[List[int]]) -> int:
        """
        -: 首先将信封的一边从小到大排序, 另一边从大到小排序(避免相同边长的情况). 随后求另一条边的最长上升子序列
        """
        envelopes.sort(key=lambda x: (x[0], -x[1]))
        return LIS([e[1]for e in envelopes])


if __name__ == "__main__":
    envelopes = [[4, 5], [4, 6], [6, 7], [2, 3], [1, 1]]
    print(Solution().maxEnvelopes(envelopes))
