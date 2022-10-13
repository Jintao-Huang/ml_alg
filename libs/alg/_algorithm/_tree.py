
from ..leetcode_utils import TreeNode
from typing import List, Optional
__all__ = ["preorder_traversal", "inorder_traversal", "postorder_traversal"]


def _preorder_traversal(root: Optional[TreeNode], res: List[int]) -> None:
    if root is None:
        return
    res.append(root.val)
    _preorder_traversal(root.left, res)
    _preorder_traversal(root.right, res)


def preorder_traversal(root: Optional[TreeNode]) -> List[int]:
    res = []
    _preorder_traversal(root, res)
    return res


def _inorder_traversal(root: Optional[TreeNode], res: List[int]) -> None:
    if root is None:
        return
    _inorder_traversal(root.left, res)
    res.append(root.val)
    _inorder_traversal(root.right, res)


def inorder_traversal(root: Optional[TreeNode]) -> List[int]:
    res = []
    _inorder_traversal(root, res)
    return res


def _postorder_traversal(root: Optional[TreeNode], res: List[int]) -> None:
    if root is None:
        return
    _postorder_traversal(root.left, res)
    _postorder_traversal(root.right, res)
    res.append(root.val)


def postorder_traversal(root: Optional[TreeNode]) -> List[int]:
    res = []
    _postorder_traversal(root, res)
    return res
