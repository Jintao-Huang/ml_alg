# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import List, Optional
try:
    from .._data_structure._ds import TreeNode
except ImportError:
    from libs.alg._data_structure._ds import TreeNode

__all__ = ["preorder_traversal", "inorder_traversal", "postorder_traversal"]


def _preorder_traversal(root: Optional[TreeNode[int]], res: List[int]) -> None:
    if root is None:
        return
    res.append(root.val)
    _preorder_traversal(root.left, res)
    _preorder_traversal(root.right, res)


def preorder_traversal(root: Optional[TreeNode[int]]) -> List[int]:
    res = []
    _preorder_traversal(root, res)
    return res


def _inorder_traversal(root: Optional[TreeNode[int]], res: List[int]) -> None:
    if root is None:
        return
    _inorder_traversal(root.left, res)
    res.append(root.val)
    _inorder_traversal(root.right, res)


def inorder_traversal(root: Optional[TreeNode[int]]) -> List[int]:
    """二叉搜索数中有用.
    Test Ref: https://leetcode.cn/problems/QO5KpG/
    """
    res = []
    _inorder_traversal(root, res)
    return res


def _postorder_traversal(root: Optional[TreeNode[int]], res: List[int]) -> None:
    if root is None:
        return
    _postorder_traversal(root.left, res)
    _postorder_traversal(root.right, res)
    res.append(root.val)


def postorder_traversal(root: Optional[TreeNode[int]]) -> List[int]:
    res = []
    _postorder_traversal(root, res)
    return res

#

def bst_min(tn: TreeNode[int]) -> int:
    assert tn is not None
    while tn.left is not None:
        tn = tn.left
    return tn.val

def bst_max(tn: TreeNode[int]) -> int:
    assert tn is not None
    while tn.right is not None:
        tn = tn.right
    return tn.val

if __name__ == "__main__":
    from libs.alg import to_tree
    tn = to_tree([6, 4, 7, 2, 5, None, 8])
    print(inorder_traversal(tn))
    print(bst_min(tn))
    print(bst_max(tn))
    