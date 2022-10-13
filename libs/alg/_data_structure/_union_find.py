
from typing import Optional


__all__ = ["UFSet"]
class UFSet:
    """同一set的根节点相同.
    -: 使用_set_size来记录每个set的数量. 只有其根节点的_set_size值是有效的.
    -: 当union时, 使用小set接在大set下的算法实现. 
        不使用树高, 因为一般树高都是2(find_root对树高进行调整).
    Test Ref: https://leetcode.cn/problems/surrounded-regions/
    """

    def __init__(self, n: int) -> None:
        self._n_set = n  # set的数量
        self.parent = [-1] * n  # -1表示是根节点
        self._set_size = [1] * n  # 每个set的元素个数.

    def find_root(self, i: int) -> int:
        """
        -: 如果i是根节点, 则返回i
            如果不是, 则递归查找parent[i], 并赋值parent[i]
        """
        if self.parent[i] == -1:
            return i

        self.parent[i] = self.find_root(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        """
        -: 如果size(ri) >= size(rj), rj接在ri下
            否则, ri接在rj下
        return: 是否union成功.
        """
        ri, rj = self.find_root(i), self.find_root(j)
        if ri == rj:
            return False

        if self._set_size[ri] >= self._set_size[rj]:
            self._set_size[ri] += self._set_size[rj]
            self.parent[rj] = ri
        else:
            self._set_size[rj] += self._set_size[ri]
            self.parent[ri] = rj
        self._n_set -= 1
        return True

    def size(self, i: Optional[int] = None) -> int:
        if i is None:
            return self._n_set
        return self._set_size[self.find_root(i)]


if __name__ == "__main__":
    ufset = UFSet(10)
    print(ufset.find_root(0))
    print(ufset.union(0, 1))
    print(ufset.union(0, 1))
    print(ufset.size())
    print(ufset.size(1))
    print(ufset.find_root(1))
