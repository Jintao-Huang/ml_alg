from typing import Callable, List, Set

__all__ = ["bs", "bs2", "n_queens"]


def bs(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    """返回满足cond的下界索引. 范围: [lo..hi]
    Test Ref: https://leetcode.cn/problems/koko-eating-bananas/
    """
    while lo < hi:
        mid = (lo + hi) // 2
        if cond(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def bs2(lo: int, hi: int, cond: Callable[[int], bool]) -> int:
    """返回满足cond的上界索引. 范围: [lo..hi]
    Test Ref: https://leetcode.cn/problems/sum-of-scores-of-built-strings/
    """
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cond(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _n_queens(n: int, i: int, path: List[int],
              v: Set[int], v2: Set[int], v3: Set[int], res: List[List[int]]) -> None:
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
        v.add(j)
        v2.add(i-j)
        v3.add(i+j)
        path.append(j)
        _n_queens(n, i + 1, path, v, v2, v3, res)
        v.remove(j)
        v2.remove(i-j)
        v3.remove(i+j)
        path.pop()


def n_queens(n: int) -> List[List[int]]:
    """返回n皇后的摆放次数
    Test Ref: https://leetcode.cn/problems/n-queens/
    """
    v = set()  # visited
    v2 = set()
    v3 = set()
    #
    res = []
    _n_queens(n, 0, [], v, v2, v3, res)
    return res


if __name__ == "__main__":
    print(n_queens(4))
