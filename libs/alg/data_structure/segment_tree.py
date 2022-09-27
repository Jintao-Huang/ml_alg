
from typing import List, Union
import math
__all__ = ["SegmentTree", "LazySegmentTree"]


class SegmentTree:
    """线段树: 不支持区间更新. 区间更新可以使用lazy线段树. 
        只支持diff. 若要支持value. 可以用存储nums, 从value计算出diff. 
    -: 使用数组存储树. 树根为0. 
    0: 存储整个段的
    """

    def __init__(self, nums: Union[List[int], int]) -> None:
        """
        -: 为什么使用h层的最大节点数存取. 因为划分区间会变成左偏树, 而不是完全二叉树.
            完全二叉树Ref: https://baike.baidu.com/item/%E5%AE%8C%E5%85%A8%E4%BA%8C%E5%8F%89%E6%A0%91/7773232  
        """
        self.n = nums if isinstance(nums, int) else len(nums)
        h = self._get_tree_height(self.n)
        # 最大节点数=2^h-1
        len_tree = (1 << h) - 1
        self.tree = [0] * len_tree
        if not isinstance(nums, int):
            self._build_tree(nums, 0, self.n-1, 0)

    @staticmethod
    def _get_tree_height(n: int) -> int:
        """1->1; 2->2; 3..4->3; 5..8->4. 最后一层的数量>=n
        -: 2^(h-1)>=n
            h>=log2(n)+1; h_min=ceil(log2(n)+1)
        """
        return math.ceil(math.log2(n) + 1)

    @staticmethod
    def _lc(i: int) -> int:
        """
        e.g. 0->1; 1->3
        """
        return (i << 1) + 1

    @staticmethod
    def _rc(i: int) -> int:
        """
        e.g. 0->2; 1->4
        """
        return (i << 1) + 2

    def _build_tree(self, nums: List[int], t_lo: int, t_hi: int, ti: int) -> int:
        """[lo..hi]
        左偏树. 后序遍历树. 
        return: 返回nums[lo..hi]的和, 用于给ti的父节点赋值.
        note: ti不会越界. 
        """
        if t_lo == t_hi:
            self.tree[ti] = nums[t_lo]
            return self.tree[ti]
        #
        mid = (t_lo + t_hi) // 2
        # [lo..mid], [mid+1, hi]. [0..2], [3..4]: 左偏树
        x1 = self._build_tree(nums, t_lo, mid, self._lc(ti))
        x2 = self._build_tree(nums, mid+1, t_hi, self._rc(ti))
        self.tree[ti] = x1 + x2
        return self.tree[ti]

    def _sum_range(self, lo: int, hi: int, t_lo: int, t_hi: int, ti: int) -> int:
        """
        -: 将求和区间不断二分, 如果与节点对应区间相同, 则直接返回值而不继续二分.
            lo..hi可能跨越mid, 可能位于一边.
        """
        if lo == t_lo and hi == t_hi:
            return self.tree[ti]
        mid = (t_lo + t_hi) // 2
        res = 0
        if lo <= mid:
            res += self._sum_range(lo, min(mid, hi), t_lo, mid, self._lc(ti))
        if hi >= mid + 1:
            res += self._sum_range(max(lo, mid + 1), hi, mid + 1, t_hi, self._rc(ti))
        return res

    def sum_range(self, lo: int, hi: int) -> int:
        assert 0 <= lo <= hi < self.n
        return self._sum_range(lo, hi, 0, self.n - 1, 0)

    def _update(self, i: int, diff: int, t_lo: int, t_hi: int, ti: int) -> None:
        """
        -: 单向树遍历, O(logn), 使用迭代法.
            进行while True循环, 直到t_lo==t_hi跳出循环. 否则重新设置t_lo, t_hi,
        """
        while True:
            self.tree[ti] += diff
            if t_lo == t_hi:
                break
            mid = (t_lo + t_hi) // 2
            if i <= mid:
                t_hi = mid
                ti = self._lc(ti)
            else:
                t_lo = mid + 1
                ti = self._rc(ti)

    def update(self, i: int, diff: int) -> None:
        assert 0 <= i < self.n
        self._update(i, diff, 0, self.n-1, 0)


if __name__ == "__main__":
    nums = [1, 2, 3, 4]
    st = SegmentTree(nums)
    print(st.tree)
    st.update(0, 1)
    print(st.tree)
    print(st.sum_range(0, 3))
    print(st.sum_range(1, 2))
    """
[10, 3, 7, 1, 2, 3, 4]
[11, 4, 7, 2, 2, 3, 4]
11
5
"""


class LazySegmentTree:
    pass
