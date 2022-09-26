

from typing import Optional, List

__all__ = ["LinkedList", "LinkedListNode"]


class LinkedListNode:
    def __init__(
        self,
        val: int,
        prev: Optional["LinkedListNode"] = None,
        next: Optional["LinkedListNode"] = None
    ) -> None:
        self.val = val
        self.prev = prev
        self.next = next


class LinkedList:
    """循环双向链表
    -: 刚开始的使用头尾两个哑节点相接
    """

    def __init__(self, nums: Optional[List[int]] = None) -> None:
        self.head = LinkedListNode(0)
        self.tail = LinkedListNode(0, self.head, self.head)
        self.head.prev = self.tail
        self.head.next = self.tail
        if nums is not None:
            self._build_list(nums)

    def _build_list(self, nums: List[int]) -> None:
        for x in nums:
            self.append(x)

    def append(self, val: int) -> None:
        self.insert_after(self.tail.prev, val)

    def append_left(self, val: int) -> None:
        self.insert_after(self.head, val)

    def pop(self):
        self.erase(self.tail.prev)

    def pop_left(self):
        self.erase(self.head.next)

    def insert_after(self, node: "LinkedListNode", val: int) -> None:
        """insert new_node after node
        -: node<->b => node<->new_node<->b
        """
        b = node.next
        new_node = LinkedListNode(val, node, b)
        node.next = new_node
        b.prev = new_node

    def erase(self, node: "LinkedListNode"):
        """
        -: a<->node<->b => a<->b
        """
        a = node.prev
        b = node.next
        a.next = b
        b.prev = a

    def __repr__(self):
        res = []
        p = self.head.next
        while p != self.tail:
            res.append(str(p.val))
            p = p.next
        return f"{self.__class__.__name__}([{', '.join(res)}])"


if __name__ == "__main__":
    from collections import deque
    l = LinkedList([1, 2, 3])
    l.append(4)
    l.pop_left()
    print(l)
    print(repr(l))
    #
    dq = deque([1, 2, 3])
    print(dq)
    print(repr(dq))
