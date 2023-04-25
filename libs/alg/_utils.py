# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Dict, List, Any, Union, TypeVar
import math
from leetcode_alg import lower_bound


def dict_sorted_key(d: Dict[int, int]) -> Dict[int, int]:
    """不通过sort value进行输出, 而是通过sort key. (not inplace)
    -: 将keys进行排序, 按k的顺序输出res. 
    """
    keys = sorted(d.keys())
    res = {}
    for k in keys:
        res[k] = d[k]
    return res


# if __name__ == "__main__":
#     print(dict_sorted_key({1: 2, 3: 4, 5: 3}))

T = TypeVar("T")


def unique(nums: List[T], keep_order: bool = True) -> List[T]:
    """not inplace. 复杂度: O(NLogN)
    -: 先获得mapper: idx(第一个)->elem. 然后将list使用set去重, 变为无序无重的list. 
        再根据mapper进行排序.  
    keep_order: 在去重的同时, 保持在list中的顺序(第一个出现的idx). 
      e.g. [2, 1, 3, 1, 2] -> [2, 1, 3]
    """
    if not keep_order:
        return list(set(nums))
    #
    idx: Dict[Any, int] = {}  # elem -> idx
    for i in reversed(range(len(nums))):
        idx[nums[i]] = i
    res = set(nums)
    return sorted(res, key=lambda x: idx[x])


# if __name__ == "__main__":
#     x = [2, 1, 3, 1, 2]
#     print(unique(x))
#     print(unique(x, keep_order=False))


def unique2(nums: List[T]) -> List[T]:
    """not inplace. 复杂度O(n). 有序数组的unique(即相同数字都聚在一起)"""
    n = len(nums)
    if n == 0:
        return []
    res = [nums[0]]
    for i in range(1, n):
        if nums[i] != nums[i-1]:
            res.append(nums[i])
    return res


# if __name__ == "__main__":
#     x = [1, 1, 2, 2, 3]
#     print(unique2(x))


def flatten_list(li: List[Union[List, int]], res: List[int]) -> None:
    """嵌套list的展平
    -: 使用树的搜索. 
    """
    for l in li:
        if isinstance(l, list):
            flatten_list(l, res)
        else:
            res.append(l)


if __name__ == "__main__":
    li = [[[1, 2], 3], 4, [5, 6, [7, 8]], 9]
    res = []
    flatten_list(li, res)
    print(res)

def soft_split(n: int, max_len: int) -> int:
    """binary search. return step
    e.g. n=21, max=10. 则n_split=ceil(21/10)=3. 则step=7. 则更均匀, 且n_split不加. 
    """
    n_split = math.ceil(n / max_len)
    return lower_bound(1, max_len, lambda mid: math.ceil(n / mid) <= n_split)

if __name__ == "__main__":
    print(soft_split(21, 10))