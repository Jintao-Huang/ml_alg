# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

from typing import List, Any, Dict

__all__ = ["unique"]


def unique(_list: List[Any], keep_order: bool = True) -> List[Any]:
    """not inplace. 复杂度: O(NLogN)
    keep_order: 在去重的同时, 保持在list中的顺序(第一个出现的idx). 
      e.g. [2, 1, 3, 1, 2] -> [2, 1, 3]
    """
    if not keep_order:
        return list(set(_list))
    #
    idx: Dict[Any, int] = {}  # elem -> idx
    for i in reversed(range(len(_list))):
        idx[_list[i]] = i
    res = set(_list)
    return sorted(res, key=lambda x: idx[x])


# if __name__ == "__main__":
#     x = [2, 1, 3, 1, 2]
#     print(unique(x))
#     print(unique(x, keep_order=False))
