# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import List, Counter, Dict, Optional, Union
try:
    from ._heapq import heapify, heappop, heappush
except ImportError:
    from libs.alg._data_structure._heapq import heapify, heappop, heappush

__all__ = ["HuffmanTree"]


class HuffmanTreeNode:
    def __init__(
        self,
        val: str,
        freq: int,
        left: Optional["HuffmanTreeNode"] = None,
        right:  Optional["HuffmanTreeNode"] = None
    ) -> None:
        self.val = val
        self.freq = freq
        self.left = left
        self.right = right


class HuffmanTree:
    def __init__(self, freq_str: Union[str, Dict[str, int]]) -> None:
        """
        freq_str: 用于计算频率的字符串
        """
        if isinstance(freq_str, str):
            self.freq: Dict[str, int] = Counter(freq_str)
        else:
            self.freq = freq_str
        assert len(self.freq) >= 2
        self.tn = self._build_tree(self.freq)
        self.mapper = {}
        self._build_mapper(self.tn, "", self.mapper)

    def _build_tree(self, freq: Dict[str, int]) -> HuffmanTreeNode:
        """
        -: 使用最普通的二叉树. 
            传入字符的频率. 然后用小根堆(森林)的特性. 每次取最小的freq对应的树进行合并.  
        """
        # k: 为了避免node比较
        heap = [(f, c, HuffmanTreeNode(c, f)) for c, f in freq.items()]
        heapify(heap)
        while len(heap) > 1:
            f, _, tn = heappop(heap)
            f2, _, tn2 = heappop(heap)
            tn_new = HuffmanTreeNode("", f + f2, tn, tn2)
            heappush(heap, (tn_new.freq, tn_new.val, tn_new))
        return heap[0][2]

    def _build_mapper(self, tn: Optional[HuffmanTreeNode], prefix: str, mapper: Dict[str, str]) -> None:
        """c -> s01. dfs(先序)"""
        if tn is None:
            return
        if tn.val != "":
            mapper[tn.val] = prefix
        #
        self._build_mapper(tn.left, prefix + "0", mapper)
        self._build_mapper(tn.right, prefix + "1", mapper)

    def encode(self, s: str) -> str:
        """返回01串"""
        res = []
        for c in s:
            res.append(self.mapper[c])
        return "".join(res)

    def decode(self, s01: str) -> str:
        """如果失败则抛出异常"""
        res = []
        n = len(s01)
        p: HuffmanTreeNode = self.tn
        for i in range(n):
            c = s01[i]
            p = p.left if c == "0" else p.right
            if p.val != "":
                res.append(p.val)
                p = self.tn
        if p is not self.tn:  # 匹配失败
            raise ValueError
        return "".join(res)


if __name__ == "__main__":
    s = "fecbda"
    freq_str = "feecccbbbbdddddaaaaaa"
    ht = HuffmanTree(freq_str)
    print(ht.freq, ht.mapper)
    s01 = ht.encode(s)
    print(s01)
    print(ht.decode(s01))
    # print(ht.decode(s01 + "0"))
