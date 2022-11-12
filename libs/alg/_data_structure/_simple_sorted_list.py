# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Optional, List, TypeVar, Any, Generic
from bisect import bisect_right, bisect_left
__all__ = ["SimpleSortedList"]
T = TypeVar("T")


class SimpleSortedList(Generic[T]):
    """
    Test Ref: https://leetcode.cn/problems/avoid-flood-in-the-city/
        当然只是为了测试. 如果为了速度, 可以使用sortedcontainers库.
    """

    def __init__(self, nums: Optional[List[T]] = None) -> None:
        if nums is None:
            nums = []
        self.sl: List[T] = nums
        self.sl.sort()

    def add(self, x: T) -> None:
        idx = self.bisect_right(x)
        self.sl.insert(idx, x)

    def remove(self, x: T) -> None:
        # 只删一个. 同sortedcontainers.SortedList
        idx = self.bisect_right(x) - 1
        if self.sl[idx] != x:
            raise ValueError("x not find")
        self.sl.pop(idx)

    def bisect_left(self, x: T) -> int:
        """return: [0..len(self)]"""
        return bisect_left(self.sl, x)

    def bisect_right(self, x: T) -> int:
        """return: [0..len(self)]"""
        return bisect_right(self.sl, x)

    def pop(self, i: int) -> T:
        return self.sl.pop(i)

    def __len__(self) -> int:
        return len(self.sl)

    def __getitem__(self, idx: int) -> T:
        return self.sl[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.sl})"


if __name__ == "__main__":
    l = [1, 1, 1]
    l.remove(1)
    print(l)
    #
    from sortedcontainers import SortedList
    sl = SortedList([1, 1, 2])
    sl.remove(1)
    print(sl)
    #
    sl = SimpleSortedList([1, 1, 2])
    sl.remove(1)
    print(sl)
