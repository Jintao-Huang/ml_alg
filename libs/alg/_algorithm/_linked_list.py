from typing import *
try:
    from .._leetcode_utils._utils import ListNode
except ImportError:
    from libs.alg._leetcode_utils._utils import ListNode

__all__ = ["find_mid_node", "reverse_list"]


def find_mid_node(head: Optional[ListNode]) -> Optional[ListNode]:
    """取下底"""
    if head is None:
        return None
    #
    p1 = head
    p2 = head  # 快指针
    while p2.next is not None and p2.next.next is not None:
        p1 = p1.next
        p2 = p2.next.next
    return p1


if __name__ == "__main__":
    from libs.alg import to_list
    ll = to_list([1, 2, 3, 4, 5])
    ll2 = to_list([1, 2, 3, 4, 5, 6])
    print(find_mid_node(ll).val)
    print(find_mid_node(ll2).val)


def reverse_list(p1: ListNode, p2: Optional[ListNode]) -> ListNode:
    """将链表反转
    -: 传入开始节点, 结束节点. 返回反转后的开始节点. (后面的节点依旧连着)
        p1 -> p -> p -> p2(None)
        将res指向p2. 然后让p1指向pe, 在让pe指向p1, p1指向下一个
    Test Ref: https://leetcode.cn/problems/reverse-linked-list-ii/
    """
    res = p2  # p_prev
    while p1 is not p2:
        pn = p1.next
        p1.next = res
        res = p1
        p1 = pn
    return res


if __name__ == "__main__":
    from libs.alg import to_list, from_list
    ll = to_list([1, 2, 3, 4, 5])
    ll = reverse_list(ll, None)
    print(from_list(ll))
