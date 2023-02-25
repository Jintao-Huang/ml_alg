from ._sort_cy import (
    quick_sort_cy, partition_cy,
    merge_cy as _merge_cy, merge_sort_cy as _merge_sort_cy
)
from numpy import ndarray
import numpy as np


def merge_cy(nums: ndarray, lo: int, mid: int, hi: int) -> None:
    """[lo..mid], [mid..hi+1]"""
    n = mid + 1 - lo
    helper = np.empty((n, ), dtype=nums.dtype)
    _merge_cy(nums, helper, lo, mid, hi)


def merge_sort_cy(nums: ndarray) -> None:
    n = (nums.shape[0] - 1) // 2 + 1
    helper = np.empty((n, ), dtype=nums.dtype)
    _merge_sort_cy(nums, helper)
