
try:
    from .._linked_list import LinkedList, LinkedListNode
except ImportError:
    from libs.alg import LinkedList, LinkedListNode
from typing import TypeVar, Dict, Tuple, Generic, Optional, Iterator

__all__ = ["MyOrderedDict"]

K = TypeVar("K")
V = TypeVar("V")


class MyOrderedDict(Generic[K, V]):
    """Test Ref: https://leetcode.cn/problems/lru-cache/"""
    def __init__(self, d: Optional[Dict[K, V]] = None) -> None:
        """
        -: 将顺序存储在list中
            self.ll: 存储Tuple[k, v]
            self.d: 存储 k -> LinkedListNode
        """
        self.ll: LinkedList[Tuple[K, V]] = LinkedList()
        self.d: Dict[K, LinkedListNode] = {}
        if isinstance(d, dict):
            self._build_od(d)

    def _build_od(self, d: Dict[K, V]) -> None:
        for k, v in d.items():
            self.append(k, v)

    def front(self) -> K:
        return self.ll.front()[0]

    def back(self) -> K:
        return self.ll.back()[0]

    def append(self, k: K, v: V, last: bool = True) -> None:
        if last:
            self.ll.append((k, v))
            ll_node = self.ll.tail.prev
        else:
            self.ll.append_left((k, v))
            ll_node = self.ll.head.next
        self.d[k] = ll_node

    def pop(self, k: K) -> V:
        ll_node = self.d.pop(k)
        v = ll_node.val[1]
        self.ll.erase(ll_node)
        return v

    def move_to_end(self, k: K, last: bool = True) -> None:
        """
        -: 通过k, 找到node. 然后删除node, 并将其插入最后会最前面. 
        """
        ll_node = self.d[k]
        k, v = ll_node.val
        self.ll.erase(ll_node)
        self.append(k, v)

    def popitem(self, last: bool = True) -> Tuple[K, V]:
        if last:
            k, v = self.ll.pop()
        else:
            k, v = self.ll.pop_left()
        self.d.pop(k)
        return k, v

    def __iter__(self) -> Iterator[K]:
        for k, _ in self.ll:
            yield k

    def __getitem__(self, k: K) -> V:
        return self.d[k].val[1]

    def __setitem__(self, k: K, v: V) -> None:
        if k in self:
            self.d[k].val = (k, v)
        else:
            self.append(k, v)

    def __delitem__(self, k: K) -> None:
        self.pop(k)

    def __repr__(self) -> str:
        res = {}
        for k in self:
            res[k] = self[k]
        return f"{self.__class__.__name__}({res})"

    def __len__(self) -> int:
        return len(self.ll)

    def __contains__(self, k: K) -> bool:
        return k in self.d

    def clear(self):
        self.ll.clear()
        self.d.clear()


if __name__ == "__main__":
    from collections import OrderedDict
    od = MyOrderedDict[int, int]({1: 1, 1: 2})
    od[1] = 3
    od[2] = 3
    od[2] = 3
    od[2] = 4
    print(od.d.keys(), od.ll)
