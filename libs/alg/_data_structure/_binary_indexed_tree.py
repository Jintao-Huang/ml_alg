# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import overload, List, Union

__all__ = ["BinaryIndexedTree"]


class BinaryIndexedTree:
    """树状数组
    -: O(logn)统计和更新前缀和/区间和.
    tree[0]=nums[0]
    tree[1]=nums[0..1]
    tree[2]=nums[2]
    tree[3]=nums[0..3]
    tree[4]=nums[4]
    tree[7]=nums[0..7]
    Test Ref: https://leetcode.cn/problems/range-sum-query-mutable/
    """

    def __init__(self, nums: Union[List[int], int]) -> None:
        """not copy
        -: 每一层末尾的0的个数都是相同的. 即lowbit(ti)相等, 覆盖范围=lowbit(ti)
        """
        # n==len(tree), 与segment tree不同.
        n = nums if isinstance(nums, int) else len(nums)
        self.tree = [0] * n
        if not isinstance(nums, int):
            self._build_tree(nums)

    @staticmethod
    def _lowbit(i: int) -> int:
        """返回二进制表示下, 最低位1及其后面0构成的数值
        e.g. lowbit(6)=lowbit(110)_2=(10)_2=2
        """
        return i & -i  # -i = ~i+1

    @classmethod
    def _parent(cls, i: int) -> int:
        """
        e.g. 0->1; 1->3; 3->7; 4->5
        """
        return i + cls._lowbit(i + 1)

    @classmethod
    def _prev(cls, i: int) -> int:
        """
        e.g. 4->3; 3->-1; 2->1
        """
        return i - cls._lowbit(i+1)

    def _build_tree(self, nums: List[int]) -> None:
        """
        -: 遍历每个nums[i]. 对于每个x, 更新其自己的节点, 以及父节点...
            只需要更新父节点即可, 而不需要更新祖先节点. 因为祖先节点将在更新父节点的时候更新. 
        """
        for i in range(len(nums)):
            self.tree[i] += nums[i]
            p = self._parent(i)
            if p < len(self.tree):
                self.tree[p] += self.tree[i]

    def prefix_sum(self, i: int) -> int:
        """[0..i]
        -:nums[0..4]=tree[4]+tree[3]
        """
        assert 0 <= i < len(self.tree)
        res = 0
        while i >= 0:
            res += self.tree[i]
            i = self._prev(i)
        return res

    def sum_range(self, lo: int, hi: int) -> int:
        """Ot(logn). [lo..hi]"""
        assert 0 <= lo < hi <= len(self.tree)
        res = self.prefix_sum(hi)
        if lo > 0:
            res -= self.prefix_sum(lo - 1)
        return res

    def update(self, i: int, diff: int) -> None:
        """只支持diff. 
            使用value更新: 可以用空间换时间的方式改进. 即存储self.nuns, update时计算diff.
        -: 将自身与祖先节点更新. Ot(logn)
        """
        assert 0 <= i < len(self.tree)
        while i < len(self.tree):
            self.tree[i] += diff
            i = self._parent(i)


if __name__ == "__main__":
    nums = [1, 2, 3, 4]
    bit = BinaryIndexedTree(nums)
    print(bit.tree)
    bit.update(0, 1)
    print(bit.tree)
    print(bit.prefix_sum(3))
    print(bit.prefix_sum(0))
    """
[1, 3, 3, 10]
[2, 4, 3, 11]
11
2
"""
