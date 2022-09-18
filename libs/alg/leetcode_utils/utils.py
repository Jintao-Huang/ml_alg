
from typing import Optional, List, Deque, Any, Dict, Tuple
from collections import deque

__all__ = ["ListNode", "TreeNode", "to_list", "from_list", "to_tree", "from_tree",
           "call_callable_list"]


class ListNode:
    def __init__(
        self,
        val: int = 0,
        next: Optional["ListNode"] = None
    ):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(
        self,
        val: int = 0,
        left: Optional["TreeNode"] = None,
        right:  Optional["TreeNode"] = None
    ) -> None:
        self.val = val
        self.left = left
        self.right = right


def to_list(li: List[int]) -> Optional[ListNode]:
    """将list(vector)转为linked-list
    思路: 对每个li中的元素, 分别加入链表中. 
    return: 若len(li)==0, 则返回None. 否则返回头结点. 
    """
    head = ListNode()
    p = head
    for x in li:
        p.next = ListNode(x)
        p = p.next
    return head.next


def from_list(ln: Optional[ListNode]) -> List[int]:
    """将linked-list转为list(vector). 
    思路: 遍历链表, 将每个元素加入list中. 并返回
    return: 若ln为None, 则返回空list.
    """
    res: List[int] = []
    while ln is not None:
        res.append(ln.val)
        ln = ln.next
    return res


if __name__ == "__main__":
    ln = to_list([1, 2, 3])
    print(from_list(ln))


def to_tree(li: List[Optional[int]]) -> Optional[TreeNode]:
    """
    思路: 将需要填的左右节点的坑放入deque. 然后按着list遍历的顺序, 不断填入坑中.
        遍历list按周期为二采取不同的行为. 
            相同行为: 创建新node, 并连接父节点. is_left置反. 新节点入dq
            若is_left=True: 则使用新的pn(弹出dq). 
    return: 若li为空, 返回None. 
    """
    if len(li) == 0:
        return None
    root = TreeNode(li[0])
    dq: Deque[TreeNode] = deque([root])
    is_left = True  # 重复True False
    for i in range(1, len(li)):
        if is_left:
            pn = dq.popleft()  # parent_node
        #
        cn = TreeNode(li[i]) if li[i] is not None else None  # child_node
        if is_left:
            pn.left = cn
        else:
            pn.right = cn
        if cn:
            dq.append(cn)
        is_left = not is_left
    return root


def _remove_last_none(res: List[Optional[int]]) -> None:
    """移除list中最后的None"""
    for i in reversed(range(len(res))):
        if res[i] is not None:
            return
        res.pop()


def from_tree(tn: Optional[TreeNode]) -> List[Optional[int]]:
    """
    思路: 使用类广搜遍历tree. 将节点.val加入res中. 
       如果某节点存在, 则将其子节点加入deque, 继续遍历. 
       如果不存在则不再遍历其子节点. 
    return: 若tn为None, 返回空list. 
    """
    res: List[Optional[int]] = []
    if tn is None:
        return res
    #
    dq: Deque[Optional[TreeNode]] = deque([tn])
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


def call_callable_list(callable_list: List[str], args_list: List[List[Any]], globals: Dict[str, Any]) -> List[Any]:
    """调用一系列可调用的函数或类. 返回可调用类/函数的返回.
    思路: 循环callable_list, args_list. 获取callable_str, args. 并从globals获取callable_obj. 并获取res
    """
    res = []
    for callable_str, args in zip(callable_list, args_list):
        callable_obj = globals[callable_str]
        res.append(callable_obj(*args))
    return res


if __name__ == "__main__":
    li = [1, None, 2, 3, None, 4, 5]
    tree = call_callable_list(["to_tree"], [[li]], globals())[0]
    li2 = call_callable_list(["from_tree"], [[tree]], globals())[0]
    print(li2)
