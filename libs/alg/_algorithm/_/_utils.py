from typing import List

__all__ = []


def reverse(nums: List[int]) -> None:
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        nums[lo], nums[hi] = nums[hi], nums[lo]
        lo += 1
        hi -= 1
    return


if __name__ == "__main__":
    nums = [1, 2, 3, 4, 5]
    reverse(nums)
    print(nums)
