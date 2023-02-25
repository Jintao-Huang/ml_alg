# distutils: language=c++
from ._types cimport *
import numpy as np

cpdef int partition_cy(int_float[::1] nums, int lo, int hi):
    cdef int_float x = nums[lo]
    while lo < hi:
        while True:  # do while
            if nums[hi] < x:
                nums[lo] = nums[hi]
                lo += 1
                break
            hi -= 1
            if lo == hi:
                break
        #
        while lo < hi:
            if nums[lo] > x:
                nums[hi] = nums[lo]
                hi -= 1
                break
            lo += 1

    nums[lo] = x
    return lo


cdef void _quick_sort_cy(int_float[::1] nums, int lo, int hi):
    if lo >= hi:
        return
    cdef int r = rand() % (hi-lo+1)+lo
    # cdef int r = (lo + hi) / 2
    nums[r], nums[lo] = nums[lo], nums[r]
    cdef int idx = partition_cy(nums, lo, hi)
    _quick_sort_cy(nums, lo, idx - 1)
    _quick_sort_cy(nums, idx + 1, hi)



def quick_sort_cy(_nums) -> None:
    srand(time(NULL))
    cdef double[::1] nums = _nums
    _quick_sort_cy(nums, 0, len(nums) - 1)



cpdef int merge_cy(int_float[::1] nums, int_float[::1] A_copy, int lo, int mid, int hi):
    # nums[lo:mid + 1]
    cdef int n = mid+1-lo
    A_copy[:n]=nums[lo:mid+1]
    cdef int i = 0, j=mid+1, k = lo
    while i < n and j <= hi:  # 避免A, B为空
        if A_copy[i] <= nums[j]:  # stable
            nums[k] = A_copy[i]
            i += 1
        else:
            nums[k] = nums[j]
            j += 1
        k += 1
    #
    while i < n:
        nums[k] = A_copy[i]
        i += 1
        k += 1


cdef void _merge_sort_cy(int_float[::1]nums, int_float[::1]A_copy, int lo, int hi):
    """[lo..hi]左偏划分
    """
    if lo == hi:
        return
    #
    cdef int mid = (lo + hi) / 2
    _merge_sort_cy(nums, A_copy, lo, mid)
    _merge_sort_cy(nums, A_copy, mid+1, hi)
    merge_cy(nums, A_copy, lo, mid, hi)


def merge_sort_cy(_nums) -> None:
    mid = _nums.shape[0] // 2
    _A_copy = np.empty((mid,), dtype=_nums.dtype)
    cdef double[::1]A_copy = _A_copy, nums = _nums
    _merge_sort_cy(nums, A_copy, 0, len(nums) - 1)

