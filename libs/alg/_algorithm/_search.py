# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Callable, List, Set, Tuple, Optional
from enum import Enum
import math
try:
    from .._data_structure._priority_queue import MutablePQ
    from .._algorithm._utils import euclidean_distance, Point
except ImportError:
    from libs.alg._data_structure._priority_queue import MutablePQ
    from libs.alg._algorithm._utils import euclidean_distance, Point

__all__ = [
    "bs", "bs2",
    "n_queens",
    "bfs_m", "greed_m", "a_star_m"
]


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


#

WALL = 1  # 不能走


def _find_path(rb: List[List[int]], ds: List[Point], s: Point, e: Point) -> List[Point]:
    res = []
    p = e
    while True:
        res.append(p)
        if p == s:
            res.reverse()
            return res
        #
        idx = rb[p[0]][p[1]]
        d = ds[idx]
        p = Point(p[0] - d[0], p[1] - d[1])


def _search_m(matrix: List[List[int]], s: Point, e: Point,
              f_cost: Callable[[Point, float], float]) -> Optional[Tuple[float, List[Point]]]:
    """查找matrix, s->t的最短路径. """
    if matrix[e[0]][e[1]] == WALL:
        return None
    #
    n, m = len(matrix), len(matrix[0])
    ds = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]  # directions
    dist: List[List[float]] = [[-1.] * m for _ in range(n)]  # 与s的距离.
    dist[s[0]][s[1]] = 0
    rb: List[List[int]] = [[-1] * m for _ in range(n)]  # rebuild
    sqrt2 = math.sqrt(2)
    #
    pq = MutablePQ[Point, float]()  # 也可以用优先级队列
    pq.add(s, f_cost(s, 0))
    while len(pq) > 0:
        v, _ = pq.pop()
        c = dist[v[0]][v[1]]
        if v == e:
            return dist[e[0]][e[1]], _find_path(rb, ds, s, e)
        #
        for i, d in enumerate(ds):
            v2 = Point(v[0] + d[0], v[1] + d[1])
            if not (0 <= v2[0] < n) or not (0 <= v2[1] < m):
                continue
            if matrix[v2[0]][v2[1]] == WALL:
                continue
            #
            c2 = c + (1 if i < 4 else sqrt2)  # cost
            if dist[v2[0]][v2[1]] != -1 and dist[v2[0]][v2[1]] <= c2:
                continue
            #
            dist[v2[0]][v2[1]] = c2
            rb[v2[0]][v2[1]] = i
            f_c = f_cost(v2, c2)
            if v2 not in pq:
                pq.add(v2, f_c)
            elif f_c < pq[v2]:
                pq.modify_priority(v2, f_c)


def bfs_m(matrix: List[List[int]], s: Point, e: Point) -> Optional[Tuple[float, List[Point]]]:
    """保证最优解"""
    return _search_m(matrix, s, e, lambda p, c: c)


def greed_m(matrix: List[List[int]], s: Point, e: Point) -> Optional[Tuple[float, List[Point]]]:
    """不保证最优解"""
    return _search_m(matrix, s, e, lambda p, c: euclidean_distance(e, p))


def a_star_m(matrix: List[List[int]], s: Point, e: Point) -> Optional[Tuple[float, List[Point]]]:
    """保证最优解. speed: greed >(优于) a_star > bfs"""
    return _search_m(matrix, s, e, lambda p, c: euclidean_distance(e, p) + c)


if __name__ == "__main__":
    env = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]
    s = Point(4, 1)
    e = Point(4, 18)
    print(bfs_m(env, s, e))
    print(greed_m(env, s, e))
    print(a_star_m(env, s, e))
