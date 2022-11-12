# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Optional, Union, List, Deque, TypeVar, Generic, Any
import json


__all__ = ["ListNode", "TreeNode", "to_list", "from_list", "to_tree", "from_tree"]


T = TypeVar("T")


class ListNode(Generic[T]):
    def __init__(
        self,
        val: T = 0,
        next: Optional["ListNode"] = None
    ):
        self.val = val
        self.next = next


class TreeNode(Generic[T]):
    def __init__(
        self,
        val: T = 0,
        left: Optional["TreeNode"] = None,
        right:  Optional["TreeNode"] = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


def to_list(li: Union[List[T], str]) -> Optional[ListNode[T]]:
    """将list(vector)转为linked-list
    思路: 对每个li中的元素, 分别加入链表中. 
    return: 若len(li)==0, 则返回None. 否则返回头结点. 
    """
    if isinstance(li, str):
        li = json.loads(li)
    assert isinstance(li, list)
    #
    head = ListNode()
    p = head
    for x in li:
        p.next = ListNode(x)
        p = p.next
    return head.next


def from_list(ln: Optional[ListNode[T]]) -> List[T]:
    """将linked-list转为list(vector). 
    思路: 遍历链表, 将每个元素加入list中. 并返回
    return: 若ln为None, 则返回空list.
    """
    res: List[T] = []
    while ln is not None:
        res.append(ln.val)
        ln = ln.next
    return res


if __name__ == "__main__":
    ln = to_list([1, 2, 3])
    print(from_list(ln))


def to_tree(li: Union[List[Optional[T]], str]) -> Optional[TreeNode[T]]:
    """
    思路: 将需要填的左右节点的坑放入deque. 然后按着list遍历的顺序, 不断填入坑中.
        遍历list按周期为二采取不同的行为. 
            相同行为: 创建新node, 并连接父节点. is_left置反. 新节点入dq
            若is_left=True: 则使用新的pn(弹出dq). 
    return: 若li为空, 返回None. 
    """
    if isinstance(li, str):
        li = json.loads(li)
    assert isinstance(li, list)
    if len(li) == 0:
        return None
    assert li[0] is not None
    #
    root = TreeNode(li[0])
    dq = Deque[TreeNode]([root])
    is_left = True  # 重复True False
    for i in range(1, len(li)):
        if is_left:
            pn = dq.popleft()  # parent_node
        #
        v = li[i]
        cn = TreeNode(v) if v is not None else None  # child_node
        if is_left:
            pn.left = cn
        else:
            pn.right = cn
        if cn:
            dq.append(cn)
        is_left = not is_left
    return root


def _remove_last_none(res: List[Optional[Any]]) -> None:
    """移除list中最后的None"""
    for i in reversed(range(len(res))):
        if res[i] is not None:
            return
        res.pop()


def from_tree(tn: Optional[TreeNode[T]]) -> List[Optional[T]]:
    """
    思路: 使用类广搜遍历tree. 将节点.val加入res中. 
       如果某节点存在, 则将其子节点加入deque, 继续遍历. 
       如果不存在则不再遍历其子节点. 
    return: 若tn为None, 返回空list. 
    """
    res: List[Optional[T]] = []
    if tn is None:
        return res
    #
    dq = Deque[Optional[TreeNode]]([tn])
    while len(dq) > 0:
        tn = dq.popleft()
        res.append(None if tn is None else tn.val)
        if tn is None:
            continue
        #
        dq.append(tn.left)
        dq.append(tn.right)
    #
    _remove_last_none(res)
    return res


if __name__ == "__main__":
    li = [1, None, 2, 3, None, 4, 5]
    tree = to_tree(li)
    print(from_tree(tree))
