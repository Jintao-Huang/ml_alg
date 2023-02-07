# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Dict, List
from numba import njit, jit
from numba.typed.typedlist import List as TypedList
from numba.typed.typeddict import Dict as TypedDict
from numpy import ndarray
from numba.core.types import ListType, int64, boolean


# 实现1, 很慢. 请使用实现2.
#   numba的没法用set(numba还没实现TypedSet), 使用字典代替.
#   TypedDict和TypedList虽然比List要快, 但是慢的要死.
@njit()
def _n_queens(n: int, i: int, path: List[int],
              v: Dict[int, bool], v2: Dict[int, bool], v3: Dict[int, bool], res: List[List[int]]) -> None:
    """
    v1-3: 分别表示: 列, 0,0点->n-1,n-1点方向的斜线. 0,n-1点->n-1,0点方向的斜线. (visited)
    """
    if i == n:
        res.append(path.copy())
        return
    #
    for j in range(n):  # 列
        if j in v or i-j in v2 or i+j in v3:
            continue
        v[j] = True
        v2[i-j] = True
        v3[i+j] = True
        path.append(j)
        _n_queens(n, i + 1, path, v, v2, v3, res)
        v.pop(j)
        v2.pop(i-j)
        v3.pop(i+j)
        path.pop()


def n_queens(n: int) -> List[List[int]]:
    """返回n皇后的摆放次数
    Test Ref: https://leetcode.cn/problems/n-queens/
    """
    v = TypedDict.empty(int64, boolean)  # visited
    v2 = TypedDict.empty(int64, boolean)
    v3 = TypedDict.empty(int64, boolean)
    #
    res = TypedList.empty_list(ListType(int64))
    path = TypedList.empty_list(int64)
    _n_queens(n, 0, path, v, v2, v3, res)
    return res

# 实现2, 使用ndarray代替set, list. 完全替代会更快, 即替代res(设置MAXSIZE).


@njit()
def _n_queens2(n: int, i: int, path: ndarray,
               v: ndarray, v2: ndarray, v3: ndarray, res: List[ndarray]) -> None:
    """
    v1-3: 分别表示: 列, 0,0点->n-1,n-1点方向的斜线. 0,n-1点->n-1,0点方向的斜线. (visited)
    """
    if i == n:
        res.append(path.copy())
        return
    #
    for j in range(n):  # 列
        if v[j] is True or v2[i-j+n - 1] is True or v3[i+j] is True:
            continue
        v[j] = True
        v2[i-j+n-1] = True
        v3[i+j] = True
        path[i] = j
        _n_queens2(n, i + 1, path, v, v2, v3, res)
        v[j] = False
        v2[i-j+n-1] = False
        v3[i+j] = False


def n_queens2(n: int) -> List[ndarray]:
    """返回n皇后的摆放次数
    Test Ref: https://leetcode.cn/problems/n-queens/
    """
    v = np.zeros((n,), dtype=np.bool8)  # visited
    v2 = np.zeros((2 * n - 1,), dtype=np.bool8)
    v3 = np.zeros((2 * n - 1,), dtype=np.bool8)
    #
    res = TypedList.empty_list(int32[:])
    path = np.empty((n,), dtype=np.int32)
    _n_queens2(n, 0, path, v, v2, v3, res)
    return res


if __name__ == "__main__":
    from libs import *
    print(n_queens(4))
    print(n_queens2(4))
    print(libs_alg.n_queens(4))
    n = 12
    libs_ml.test_time(lambda: n_queens(n), warm_up=1)
    libs_ml.test_time(lambda: n_queens2(n), warm_up=1)
    libs_ml.test_time(lambda: libs_alg.n_queens(n), warm_up=1)
