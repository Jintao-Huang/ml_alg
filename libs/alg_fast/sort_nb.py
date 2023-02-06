from numba import njit, guvectorize
from numba.typed.typedlist import List as TypedList
from numba.core.types import void, int64, float64, ListType, List as ReflectList, Array
from numba import typeof
import random

__all__ = ["partition", "quick_sort", "merge", "merge_sort"]

# int64(ListType(int64), int64, int64)
@njit()
def partition(nums, lo, hi):
    """
    -: 将nums[lo]为标杆, 将小于的放置其右边, 大于的放置其左. 等于的随意. 
        使用相撞二指针实现. 当lo == hi时跳出循环. 
        循环中, 先遍历hi那边, 不断下降, 直到等于lo或nums[hi]小于x, 则将值放置lo(空缺处). 此时hi空缺.
        再遍历lo这边, 不断上升, 直到等于hi或nums[lo]大于x, 则将值放置到hi, 此时lo空缺
    """
    x = nums[lo]
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


if __name__ == "__main__":
    nums = TypedList([1, 4, 6, 3, 2, 5, 0, 8, 9, 7])
    print(partition(nums, 0, len(nums) - 1))
    print(nums)


# void(ListType(int64), int64, int64)
@njit()
def _quick_sort(nums, lo, hi):
    """[lo..hi]. 右偏划分
    -: 划分法. 使用随机算法
    """
    if lo >= hi:
        return
    r = random.randint(lo, hi)
    nums[r], nums[lo] = nums[lo], nums[r]
    idx = partition(nums, lo, hi)
    _quick_sort(nums, lo, idx - 1)
    _quick_sort(nums, idx + 1, hi)


# void(ListType(int64))
@njit()
def quick_sort(nums):
    _quick_sort(nums, 0, len(nums) - 1)


if __name__ == "__main__":
    nums = TypedList([1, 4, 6, 3, 2, 5, 0, 8, 9, 7])
    quick_sort(nums)
    print(nums)


# void(ListType(int64), int64, int64, int64)
@njit()
def merge(nums, lo, mid, hi):
    """merge nums[lo..mid], nums[mid+1..hi]
    -: A: nums[lo..mid]和B: nums[mid+1..hi]都是顺序的(小->大)
        我们创建一个A_copy, 随后不断将A_copy和B的较小值, 放入nums中.
    #
        如果有一个放完后, 我们将A_copy剩余的放入nums中.
    """
    A_copy = nums[lo:mid + 1]  # 浅复制
    i, j, k = 0, mid + 1, lo
    n = len(A_copy)
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


if __name__ == "__main__":
    nums = TypedList([1, 3, 4, 7, 9, 0, 2, 5, 6, 8])
    merge(nums, 0, 4, 9)
    print(nums)
    nums = TypedList([3, 1])
    merge(nums, 0, 0, 1)
    print(nums)
    nums = TypedList([3])
    merge(nums, 0, 0, 0)
    print(nums)


# void(ListType(int64), int64, int64)
@njit()
def _merge_sort(nums, lo, hi):
    """[lo..hi]左偏划分
    """
    if lo == hi:
        return
    #
    mid = (lo + hi) // 2
    _merge_sort(nums, lo, mid)
    _merge_sort(nums, mid+1, hi)
    merge(nums, lo, mid, hi)


# void(ListType(int64))
@njit()
def merge_sort(nums):
    _merge_sort(nums, 0, len(nums) - 1)


if __name__ == "__main__":
    nums = TypedList([1, 4, 6, 3, 2, 5, 0, 8, 9, 7])
    merge_sort(nums)
    print(nums)


@guvectorize("(n)->(n)")
def quick_sort_np(nums, res):
    res[:] = nums
    quick_sort(res)


if __name__ == "__main__":
    # 快 < ndarray.sort < list.sort < quick_sort(x） < quick_sort(x_tl)
    #   < quick_sort(x_l) < python_quick_sort(x_l) < 慢
    from libs import *
    x = np.random.randn(1000000)
    print()
    x_l = libs_ml.test_time(lambda: x.tolist())
    x_tl = libs_ml.test_time(lambda: TypedList(x))  # very slow
    libs_ml.test_time(lambda: np.sort(x), 1)  # fast
    libs_ml.test_time(lambda: x_l.sort(), 1)
    libs_ml.test_time(lambda: quick_sort(x.copy()), 1, warm_up=1)  # fast
    libs_ml.test_time(lambda: quick_sort(x_tl), 1, warm_up=1)
    # libs_ml.test_time(lambda: quick_sort(x_l), 1)  # warning
    libs_ml.test_time(lambda: libs_alg.quick_sort(x_l), 1)  # very slow
    print()
    res = np.empty_like(x)
    libs_ml.test_time(lambda: quick_sort_np(x, res), 1, warm_up=1)  # fast
    x_2d = x.reshape(10, 100000)
    res = np.empty_like(x_2d)
    libs_ml.test_time(lambda: quick_sort_np(x_2d, res), 1, warm_up=1)  # fast
    print()


# def heap_sort():
#     """见`data_structure/heapq_.py`"""
#     ...
