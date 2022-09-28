
import random
from typing import TypeVar, List, Any
from copy import copy
from heapq import heapify, heappop

__all__ = []
T = TypeVar("T", float, int)


def partition(arr: List[T], lo: int, hi: int) -> int:
    """
    -: 将arr[lo]为标杆, 将小于的放置其右边, 大于的放置其左. 等于的随意. 
        使用相撞二指针实现. 当lo == hi时跳出循环. 
        循环中, 先遍历hi那边, 不断下降, 直到等于lo或arr[hi]小于x, 则将值放置lo(空缺处). 此时hi空缺.
        再遍历lo这边, 不断上升, 直到等于hi或arr[lo]大于x, 则将值放置到hi, 此时lo空缺
    """
    x = arr[lo]
    while lo < hi:
        while True:  # do while
            if arr[hi] < x:
                arr[lo] = arr[hi]
                lo += 1
                break
            hi -= 1
            if lo == hi:
                break
        #
        while lo < hi:
            if arr[lo] > x:
                arr[hi] = arr[lo]
                hi -= 1
                break
            lo += 1

    arr[lo] = x
    return lo


if __name__ == "__main__":
    arr = [1, 4, 6, 3, 2, 5, 0, 8, 9, 7]
    print(partition(arr, 0, len(arr) - 1))
    print(arr)


def _quick_sort(arr: List[T], lo: int, hi: int) -> None:
    """[lo..hi]. 右偏划分
    -: 划分法. 使用随机算法
    """
    if lo >= hi:
        return
    r = random.randint(lo, hi)
    arr[r], arr[lo] = arr[lo], arr[r]
    idx = partition(arr, lo, hi)
    _quick_sort(arr, lo, idx - 1)
    _quick_sort(arr, idx + 1, hi)


def quick_sort(arr: List[T]) -> None:
    _quick_sort(arr, 0, len(arr) - 1)


if __name__ == "__main__":
    arr = [1, 4, 6, 3, 2, 5, 0, 8, 9, 7]
    quick_sort(arr)
    print(arr)


def merge(arr: List[T], lo: int, mid: int, hi: int) -> None:
    """merge arr[lo..mid], arr[mid+1..hi]
    -: A:arr[lo..mid]和B:arr[mid+1..hi]都是顺序的(小->大)
        我们创建一个A_copy, 随后不断将A_copy和B的较小值, 放入arr中. 
    # 
        如果有一个放完后, 我们将A_copy剩余的放入arr中.
    """
    A_copy = copy(arr[lo:mid + 1])
    i, j, k = 0, mid + 1, lo
    n = len(A_copy)
    while i < n and j <= hi:  # 避免A, B为空
        if A_copy[i] <= arr[j]:  # stable
            arr[k] = A_copy[i]
            i += 1
        else:
            arr[k] = arr[j]
            j += 1
        k += 1
    #
    while i < n:
        arr[k] = A_copy[i]
        i += 1
        k += 1


if __name__ == "__main__":
    arr = [1, 3, 4, 7, 9, 0, 2, 5, 6, 8]
    merge(arr, 0, 4, 9)
    print(arr)
    arr = [3, 1]
    merge(arr, 0, 0, 1)
    print(arr)
    arr = [3]
    merge(arr, 0, 0, 0)
    print(arr)


def _merge_sort(arr: List[T], lo: int, hi: int) -> None:
    """[lo..hi]左偏划分
    """
    if lo == hi:
        return
    #
    mid = (lo + hi) // 2
    _merge_sort(arr, lo, mid)
    _merge_sort(arr, mid+1, hi)
    merge(arr, lo, mid, hi)


def merge_sort(arr: List[T]) -> None:
    _merge_sort(arr, 0, len(arr) - 1)


if __name__ == "__main__":
    arr = [1, 4, 6, 3, 2, 5, 0, 8, 9, 7]
    merge_sort(arr)
    print(arr)


def heap_sort(arr: List[T], dst: List[T]) -> None:
    """处理后arr将成为空列表
    -: 迭代法. 不是inplace, 但空间复杂度O(1)
    -: 先用小根堆, 不断产生最小值, 然后不断加入dst中. 
    Note: inplace的算法见`heapq_.py`
    """
    heapify(arr)
    for _ in range(len(arr)):
        dst.append(heappop(arr))
    

if __name__ == "__main__":
    arr = [1, 4, 6, 3, 2, 5, 0, 8, 9, 7]
    dst = []
    heap_sort(arr, dst)
    print(arr)
    print(dst)