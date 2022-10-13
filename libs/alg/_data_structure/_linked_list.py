

from typing import Optional, List, TypeVar, Generic, Iterator

__all__ = ["LinkedList", "LinkedListNode"]

T = TypeVar("T")


class LinkedListNode(Generic[T]):
    def __init__(
        self,
        val: T,
        prev: "LinkedListNode",
        next: "LinkedListNode"
    ) -> None:
        self.val = val
        self.prev = prev
        self.next = next


class LinkedList(Generic[T]):
    """循环双向链表
    -: 刚开始的使用头尾两个哑节点相接
    Test Ref: MyOrderedDict: https://leetcode.cn/problems/lru-cache/
    """

    def __init__(self, nums: Optional[List[T]] = None) -> None:
        self.head = LinkedListNode(0, None, None)
        self.tail = LinkedListNode(0, self.head, self.head)
        self.head.prev = self.tail
        self.head.next = self.tail
        self._len = 0
        if nums is not None:
            self._build_list(nums)

    def _build_list(self, nums: List[T]) -> None:
        for x in nums:
            self.append(x)

    def append(self, val: T) -> None:
        self.insert_after(self.tail.prev, val)

    def append_left(self, val: T) -> None:
        self.insert_after(self.head, val)

    def pop(self) -> T:
        res = self.back()
        self.erase(self.tail.prev)
        return res

    def pop_left(self) -> T:
        res = self.front()
        self.erase(self.head.next)
        return res

    def _getitem(self, node: "LinkedListNode") -> T:
        assert node is not self.head and node is not self.tail
        return node.val

    def front(self) -> T:
        return self._getitem(self.head.next)

    def back(self) -> T:
        return self._getitem(self.tail.prev)

    def insert_after(self, node: "LinkedListNode", val: T) -> None:
        """insert new_node after node
        -: node<->b => node<->new_node<->b
        """
        b = node.next
        new_node = LinkedListNode(val, node, b)
        node.next = new_node
        b.prev = new_node
        self._len += 1

    def erase(self, node: "LinkedListNode") -> None:
        """
        -: a<->node<->b => a<->b
        """
        assert self._len > 0
        a = node.prev
        b = node.next
        a.next = b
        b.prev = a
        self._len -= 1

    def __iter__(self) -> Iterator[T]:
        p = self.head.next
        while p != self.tail:
            yield p.val
            p = p.next

    def __repr__(self) -> str:
        res = []
        for x in self:
            res.append(x)
        return f"{self.__class__.__name__}({res})"

    def __len__(self) -> int:
        return self._len

    def clear(self):
        self.head.prev = self.tail
        self.head.next = self.tail
        self.tail.prev = self.head
        self.tail.next = self.head
        self._len = 0


if __name__ == "__main__":
    from collections import deque
    l = LinkedList([1, 2, 3])
    l.append(4)
    l.pop_left()
    print(l)
    print(l.front())
    l.clear()
    print(l)
    print(len(l))
    """
LinkedList([2, 3, 4])
2
4
4
LinkedList([])
0
"""
