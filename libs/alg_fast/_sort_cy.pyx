# distutils: language=c++
# distutils: extra_compile_args = ["-std=c++11"]
from ._types cimport *

cdef int _partition_cy(int_float[::1] nums, int lo, int hi):
    # [lo..hi]
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

def partition_cy(int_float[::1] nums) -> int:
    return _partition_cy(nums, 0, <int>nums.shape[0] - 1)

cdef void _quick_sort_cy(int_float[::1] nums, int lo, int hi):
    # [lo..hi]
    if lo >= hi:
        return
    cdef int r = rand() % (hi-lo+1)+lo
    # cdef int r = (lo + hi) / 2
    nums[r], nums[lo] = nums[lo], nums[r]
    cdef int idx = _partition_cy(nums, lo, hi)
    _quick_sort_cy(nums, lo, idx - 1)
    _quick_sort_cy(nums, idx + 1, hi)



def quick_sort_cy(int_float[::1] nums) -> None:
    srand(time(NULL))
    _quick_sort_cy(nums, 0, len(nums) - 1)



cdef void _merge_cy(int_float[::1] nums, int_float[::1] helper, int lo, int mid, int hi):
    # [lo..mid]; [mid+1..hi]
    cdef int n = mid+1-lo
    helper[:n]=nums[lo:mid+1]
    cdef int i = 0, j=mid+1, k = lo
    while i < n and j <= hi:  # 避免A, B为空
        if helper[i] <= nums[j]:  # stable
            nums[k] = helper[i]
            i += 1
        else:
            nums[k] = nums[j]
            j += 1
        k += 1
    #
    while i < n:
        nums[k] = helper[i]
        i += 1
        k += 1

def merge_cy(int_float[::1]nums, int_float[::1] helper, int lo, int mid, int hi) -> None:
    # [lo..mid]; [mid+1..hi]
    # helper.shape >= mid+1-lo
    _merge_cy(nums, helper, lo, mid, hi)


cdef void _merge_sort_cy(int_float[::1]nums, int_float[::1]helper, int lo, int hi):
    """[lo..hi]左偏(多)划分"""
    if lo == hi:
        return
    #
    cdef int mid = (lo + hi) / 2
    _merge_sort_cy(nums, helper, lo, mid)
    _merge_sort_cy(nums, helper, mid+1, hi)
    _merge_cy(nums, helper, lo, mid, hi)


def merge_sort_cy(int_float[::1] nums, int_float[::1] helper) -> None:
    # helper.shape >= (nums.shape[0]-1)//2+1
    _merge_sort_cy(nums, helper, 0, <int>nums.shape[0] - 1)
