"""test BinaryIndexedTree, SegmentTree"""
from libs import *
from libs.alg import *


class NumArray:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.bit = BinaryIndexedTree(nums)

    def update(self, index: int, val: int) -> None:
        diff = val - self.nums[index]
        self.nums[index] = val
        self.bit.update(index, diff)

    def sumRange(self, left: int, right: int) -> int:
        return self.bit.sum_range(left, right)


class NumArray2:
    def __init__(self, nums: List[int]):
        self.nums = nums
        self.st = SegmentTree(nums)

    def update(self, index: int, val: int) -> None:
        diff = val - self.nums[index]
        self.nums[index] = val
        self.st.update(index, diff)

    def sumRange(self, left: int, right: int) -> int:
        return self.st.sum_range(left, right)


if __name__ == "__main__":
    print(call_callable_list(["NumArray", "sumRange", "update", "sumRange"],
                             [[[1, 3, 5]], [0, 2], [1, 2], [0, 2]], globals()))
    print(call_callable_list(["NumArray2", "sumRange", "update", "sumRange"],
                             [[[1, 3, 5]], [0, 2], [1, 2], [0, 2]], globals()))
