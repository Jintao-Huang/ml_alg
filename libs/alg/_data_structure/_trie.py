# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Optional, Dict

__all__ = ["Trie"]


class TrieTreeNode:
    def __init__(self, val: str) -> None:
        self.val = val
        self.finish = False
        self.children: Dict[str, "TrieTreeNode"] = {}


class Trie:
    """
    -: 多叉树
    Test Ref: https://leetcode.cn/problems/implement-trie-prefix-tree/
    """

    def __init__(self) -> None:
        # "root为空节点"
        self.root = TrieTreeNode("\0")

    def insert(self, s: str) -> None:
        """迭代法
        -: 从根节点, 不断遍历s. 如果节点不存在, 则创建节点, 并修改父节点. 
            最后的节点再修改finish
        """
        p = self.root
        for c in s:
            if c not in p.children:
                p.children[c] = TrieTreeNode(c)
            p = p.children[c]
        p.finish = True

    def _search_prefix(self, prefix: str) -> Optional[TrieTreeNode]:
        p = self.root
        for c in prefix:
            if c not in p.children:
                return None
            p = p.children[c]
        return p

    def starts_with(self, prefix: str) -> bool:
        return self._search_prefix(prefix) is not None

    def search(self, s: str) -> bool:
        p = self._search_prefix(s)
        return p is not None and p.finish
