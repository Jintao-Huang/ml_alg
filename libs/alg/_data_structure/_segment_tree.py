
from typing import List, Union
import math
try:
    from ._ import _lc
except ImportError:  # for debug
    from _ import _lc
__all__ = ["SegmentTree", "LazySegmentTree"]


class SegmentTree:
    """线段树(左偏树): 不支持区间更新. 区间更新可以使用lazy线段树. 
        只支持diff. 若要支持value. 可以用存储nums, 从value计算出diff.
    -: 使用数组存储树. 树根为0.
        self.tree[0]: 存储整个段的: 树根. (与树状数组不同)
    Test Ref: https://leetcode.cn/problems/range-sum-query-mutable/
    """

    def __init__(self, nums: Union[List[int], int]) -> None:
        """
        -: 为什么使用h层的最大节点数存取. 因为划分区间会变成左偏树, 而不是完全二叉树.
            完全二叉树Ref: https://baike.baidu.com/item/%E5%AE%8C%E5%85%A8%E4%BA%8C%E5%8F%89%E6%A0%91/7773232
        """
        self.n = nums if isinstance(nums, int) else len(nums)
        h = self._get_tree_height(self.n)
        # 最大节点数=2^h-1
        tree_len = (1 << h) - 1
        self.tree = [0] * tree_len
        if not isinstance(nums, int):
            self._build_tree(nums, 0, self.n-1, 0)

    @staticmethod
    def _get_tree_height(n: int) -> int:
        """1->1; 2->2; 3..4->3; 5..8->4. 最后一层的数量>=n
        -: 2^(h-1)>=n
            h>=log2(n)+1; h_min=ceil(log2(n)+1)
        """
        return math.ceil(math.log2(n) + 1)

    def _build_tree(self, nums: List[int], t_lo: int, t_hi: int, ti: int) -> int:
        """[t_lo..t_hi]
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
        lc = _lc(ti)
        rc = lc + 1
        x1 = self._build_tree(nums, t_lo, mid, lc)
        x2 = self._build_tree(nums, mid+1, t_hi, rc)
        self.tree[ti] = x1 + x2
        return self.tree[ti]

    def _sum_range(self, lo: int, hi: int, t_lo: int, t_hi: int, ti: int) -> int:
        """[lo..hi]; t_lo..t_hi]
        -: 将求和区间不断二分, 如果与节点对应区间相同, 则直接返回值而不继续二分.
            lo..hi可能跨越mid, 可能位于一边.
        """
        if lo == t_lo and hi == t_hi:
            return self.tree[ti]
        mid = (t_lo + t_hi) // 2
        res = 0
        lc = _lc(ti)
        if lo <= mid:
            res += self._sum_range(lo, min(mid, hi), t_lo, mid, lc)
        if hi >= mid + 1:
            res += self._sum_range(max(lo, mid + 1), hi, mid + 1, t_hi, lc + 1)
        return res

    def sum_range(self, lo: int, hi: int) -> int:
        assert 0 <= lo <= hi < self.n
        return self._sum_range(lo, hi, 0, self.n - 1, 0)

    def _update(self, i: int, diff: int, t_lo: int, t_hi: int, ti: int) -> None:
        """[t_lo..t_hi]
        -: 单向树遍历, O(logn), 使用迭代法.
            进行while True循环, 直到t_lo==t_hi跳出循环. 否则重新设置t_lo, t_hi,
        """
        while True:
            self.tree[ti] += diff
            if t_lo == t_hi:
                break
            mid = (t_lo + t_hi) // 2
            lc = _lc(ti)
            if i <= mid:
                t_hi = mid
                ti = lc
            else:
                t_lo = mid + 1
                ti = lc + 1

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
    print()
    """
[10, 3, 7, 1, 2, 3, 4]
[11, 4, 7, 2, 2, 3, 4]
11
5
"""


class LazySegmentTree:
    """(左偏树)
    Test Ref(diff_style=False): https://leetcode.cn/problems/QO5KpG/
        python会超时, 这里只是为了验证算法正确性. cpp下可以通过.
    """

    def __init__(self, nums: Union[List[int], int], diff_style: bool = True) -> None:
        """
        lazy_tag[i]: 表示以i为树根的子树. 树根已更新, 但左右树未更新.
        """
        self.n = nums if isinstance(nums, int) else len(nums)
        self.diff_style = diff_style
        h = self._get_tree_height(self.n)
        # 最大节点数=2^h-1
        tree_len = (1 << h) - 1
        tag_len = (1 << (h - 1)) - 1
        self.tree = [0] * tree_len
        self.lazy_tag = [0 if diff_style else None] * tag_len
        if not isinstance(nums, int):
            self._build_tree(nums, 0, self.n-1, 0)

    @staticmethod
    def _get_tree_height(n: int) -> int:
        """n表示树叶子节点的数量"""
        return math.ceil(math.log2(n) + 1)

    def _build_tree(self, nums: List[int], t_lo: int, t_hi: int, ti: int) -> int:
        if t_lo == t_hi:
            self.tree[ti] = nums[t_lo]
            return self.tree[ti]
        #
        mid = (t_lo + t_hi) // 2
        # [lo..mid], [mid+1, hi]. [0..2], [3..4]: 左偏树
        lc = _lc(ti)
        rc = lc + 1
        x1 = self._build_tree(nums, t_lo, mid, lc)
        x2 = self._build_tree(nums, mid+1, t_hi, rc)
        self.tree[ti] = x1 + x2
        return self.tree[ti]

    def _update_lazy_tag(self, t_lo: int, t_hi: int, ti: int) -> None:
        # lazy tag
        lc = _lc(ti)
        mid = (t_lo + t_hi) // 2
        lazyt = self.lazy_tag[ti]
        # 
        if self.diff_style:
            assert lazyt is not None
            if lazyt != 0:
                self.lazy_tag[ti] = 0
                self.tree[lc] += (mid - t_lo + 1) * lazyt
                self.tree[lc + 1] += (t_hi - mid) * lazyt
                if lc < len(self.lazy_tag):
                    self.lazy_tag[lc] = lazyt
                    self.lazy_tag[lc + 1] = lazyt
        else:
            if lazyt is not None:
                self.lazy_tag[ti] = None
                self.tree[lc] = (mid - t_lo + 1) * lazyt
                self.tree[lc + 1] = (t_hi - mid) * lazyt
                if lc < len(self.lazy_tag):
                    self.lazy_tag[lc] = lazyt
                    self.lazy_tag[lc + 1] = lazyt

    def _sum_range(self, lo: int, hi: int, t_lo: int, t_hi: int, ti: int) -> int:
        """lazy_tag[ti] -> lazy_tag[lc;rc] tree[lc;rc]
        -: lazy_tag表示以该节点为树根的左右子树未进行更新(note: 树根已更新, 左右子树未更新). 
            在sum_range时, 将可以更新的lazy_tag进行顺便更新.
        """
        if lo == t_lo and hi == t_hi:
            return self.tree[ti]
        
        #
        self._update_lazy_tag(t_lo, t_hi, ti)
        lc = _lc(ti)
        mid = (t_lo + t_hi) // 2
        res = 0
        if lo <= mid:
            res += self._sum_range(lo, min(mid, hi), t_lo, mid, lc)
        if hi >= mid + 1:
            res += self._sum_range(max(lo, mid + 1), hi, mid + 1, t_hi, lc + 1)
        return res

    def sum_range(self, lo: int, hi: int) -> int:

        assert 0 <= lo <= hi < self.n
        return self._sum_range(lo, hi, 0, self.n - 1, 0)

    def _update(self, lo: int, hi: int, d_v: int, t_lo: int, t_hi: int, ti: int) -> None:
        """更新tree[ti], lazy_tag[ti].
        -: 将lo, hi进行分割, 直到lo, hi修改完所有的lazy tag. 并顺便更新lazy_tag
        """
        if lo == t_lo and hi == t_hi:
            if ti < len(self.lazy_tag):
                self.lazy_tag[ti] = d_v
            # 
            if self.diff_style:
                self.tree[ti] += (t_hi - t_lo + 1) * d_v
            else:
                self.tree[ti] = (t_hi - t_lo + 1) * d_v
            return
        #
        self._update_lazy_tag(t_lo, t_hi, ti)
        lc = _lc(ti)
        mid = (t_lo + t_hi) // 2
        if lo <= mid:
            self._update(lo, min(mid, hi), d_v, t_lo, mid, lc)
        if hi >= mid + 1:
            self._update(max(lo, mid + 1), hi, d_v, mid + 1, t_hi, lc + 1)
        self.tree[ti] = self.tree[lc] + self.tree[lc + 1]



    def update(self, lo: int, hi: int, diff: int) -> None:
        assert 0 <= lo <= hi < self.n
        self._update(lo, hi, diff, 0, self.n-1, 0)


if __name__ == "__main__":
    nums = [1, 2, 3, 4]
    st = LazySegmentTree(nums, True)
    print(st.tree)
    print(st.lazy_tag)
    st.update(0, 2, 1)
    print(st.tree)
    print(st.lazy_tag)
    print(st.sum_range(0, 3))
    print(st.tree)
    print(st.lazy_tag)
    print(st.sum_range(1, 2))
    print(st.tree)
    print(st.lazy_tag)
    #
    print()
    nums = [1, 2, 3, 4]
    st = LazySegmentTree(nums, False)
    print(st.tree)
    print(st.lazy_tag)
    st.update(0, 2, 1)
    print(st.tree)
    print(st.lazy_tag)
    print(st.sum_range(0, 3))
    print(st.tree)
    print(st.lazy_tag)
    print(st.sum_range(1, 2))
    print(st.tree)
    print(st.lazy_tag)
