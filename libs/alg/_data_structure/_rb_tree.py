from typing import Generic, TypeVar, List, Optional
from enum import Enum


__all__ = ["RBSortedList"]


class Color(Enum):
    Red = 1
    Black = 2


T = TypeVar("T")


class RBTreeNode(Generic[T]):
    def __init__(self, key: T, color: Color = Color.Black, size: int = 0,
                 p: Optional["RBTreeNode"] = None, left: Optional["RBTreeNode"] = None, right: Optional["RBTreeNode"] = None):
        self.key = key
        self.color = color
        # x.size = x.left.size + x.right.size + 1
        self.size = size  # 树的大小.
        self.p: "RBTreeNode" = p
        self.left: "RBTreeNode" = left
        self.right: "RBTreeNode" = right


"""红黑树特性
1. 特殊的二叉搜索树(BST). 保证没有一条路径会比其他路径长2倍, 因此是平衡的.
2. 若某节点无子节点/父节点, 则对应指针指向nil(用nil节点代替None, 使得代码结构更清晰)
    空树: root=nil. 
3. 根节点, nil为黑色. 若某节点为红, 则取两节点为黑. 
    对于每个节点, 该节点到nil的路径上, 具有相同黑色节点. 
"""


class RBTree(Generic[T]):
    """
    Ref: 算法导论
    note: 可重复. 类似于c++的multiset
    """

    def __init__(self):
        # 空树
        nil = RBTreeNode(0, Color.Black)
        self.root = nil
        self.nil = nil

    def getitem(self, i: int) -> RBTreeNode:
        return self._getitem(self.root, i)

    def _getitem(self, tn: RBTreeNode, i: int) -> RBTreeNode:
        """返回x为根的子树的第i小关键字(从0开始)(即SortedList中sl[i])的节点"""
        idx = tn.left.size  # 当前节点的索引
        if i == idx:
            return tn
        elif i < idx:
            return self._getitem(tn.left, i)
        else:
            return self._getitem(tn.right, i - idx - 1)

    def bisect_left(self, x: T) -> int:
        """获取SortedList中x值的lower bound索引."""
        tn = self.root
        res = 0
        while tn is not self.nil:
            if x <= tn.key:
                tn = tn.left
            else:
                res += tn.left.size + 1
                tn = tn.right
        return res

    def bisect_right(self, x: T) -> int:
        """获取SortedList中x值的upper bound索引."""
        tn = self.root
        res = 0
        while tn is not self.nil:
            if x < tn.key:
                tn = tn.left
            else:
                res += tn.left.size + 1
                tn = tn.right
        return res

    def search(self, x: T) -> RBTreeNode:
        """搜索key=k的tn"""
        tn = self.root
        while tn is not self.nil and x != tn.key:
            if x < tn.key:
                tn = tn.left
            else:
                tn = tn.right
        return tn

    def _left_rotate(self, x: RBTreeNode) -> None:
        """左旋. x为旋转根"""
        y = x.right
        assert y is not self.nil  # 假设
        x.right = y.left
        if y.left is not self.nil:
            y.left.p = x
        y.p = x.p
        if x.p is self.nil:
            self.root = y
        elif x is x.p.left:
            x.p.left = y
        else:
            x.p.right = y
        y.left = x
        x.p = y
        y.size = x.size
        x.size = x.left.size + x.right.size + 1

    def _right_rotate(self, y: RBTreeNode) -> None:
        """右旋, y为旋转根"""
        x = y.left
        assert x is not self.nil
        y.left = x.right
        if x.right is not self.nil:
            x.right.p = y
        x.p = y.p
        if y.p is self.nil:
            self.root = x
        elif y is y.p.left:
            y.p.left = x
        else:
            y.p.right = x
        x.right = y
        y.p = x
        x.size = y.size
        y.size = y.left.size + y.right.size + 1

    def insert(self, z: RBTreeNode) -> None:
        """插入z节点"""
        y = self.nil
        x = self.root
        # 找位置.
        while x is not self.nil:
            y = x
            x.size += 1
            if z.key < x.key:
                x = x.left
            else:
                x = x.right
        # 插入
        z.p = y
        if y is self.nil:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.nil
        z.right = self.nil
        z.color = Color.Red  # 置为红, z可能违背红黑性质
        z.size = 1
        self._rb_insert_fixup(z)

    def _rb_insert_fixup(self, z: RBTreeNode) -> None:
        # z: 可能违背红黑性质的节点
        while z.p.color is Color.Red:
            if z.p is z.p.p.left:
                y = z.p.p.right
                if y.color is Color.Red:
                    z.p.color = Color.Black
                    y.color = Color.Black
                    z.p.p.color = Color.Red
                    z = z.p.p
                elif z is z.p.right:
                    z = z.p
                    self._left_rotate(z)
                else:
                    z.p.color = Color.Black
                    z.p.p.color = Color.Red
                    self._right_rotate(z.p.p)
            else:
                y = z.p.p.left
                if y.color is Color.Red:
                    z.p.color = Color.Black
                    y.color = Color.Black
                    z.p.p.color = Color.Red
                    z = z.p.p
                elif z is z.p.left:
                    z = z.p
                    self._right_rotate(z)
                else:
                    z.p.color = Color.Black
                    z.p.p.color = Color.Red
                    self._left_rotate(z.p.p)
        self.root.color = Color.Black

    def _rb_transplant(self, u: RBTreeNode, v: RBTreeNode) -> None:
        # 用v替代u, 删除节点时使用
        if u.p is self.nil:
            self.root = v
        elif u is u.p.left:
            u.p.left = v
        else:
            u.p.right = v
        v.p = u.p

    def _tree_minimum(self, x: RBTreeNode) -> RBTreeNode:
        """找二叉搜索树中的最小值(一直往左找)"""
        while x.left is not self.nil:
            x = x.left
        return x

    def delete(self, z: RBTreeNode) -> None:
        """删除z"""
        # y: 在树中删除的节点或者移至树内的节点
        # x: 移到y的原始位置上
        y = z
        y_original_color = y.color
        if z.left is self.nil:
            x = z.right
            self._rb_transplant(z, z.right)
        elif z.right is self.nil:
            x = z.left
            self._rb_transplant(z, z.left)
        else:
            y = self._tree_minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.p is z:
                x.p = y
            else:
                self._rb_transplant(y, y.right)
                y.right = z.right
                y.right.p = y
            self._rb_transplant(z, y)
            y.left = z.left
            y.left.p = y
            y.color = z.color
            y.size = z.size
        p = x.p
        while p is not self.nil:
            p.size -= 1
            p = p.p
        #
        if y_original_color is Color.Black:
            self._rb_delete_fixup(x)  # 删了一个黑

    def _rb_delete_fixup(self, x: RBTreeNode) -> None:
        while x is not self.root and x.color is Color.Black:
            if x is x.p.left:
                w = x.p.right
                if w.color is Color.Red:
                    w.color = Color.Black
                    x.p.color = Color.Red
                    self._left_rotate(x.p)
                    w = x.p.right
                if w.left.color is Color.Black and w.right.color is Color.Black:
                    w.color = Color.Red
                    x = x.p
                elif w.right.color is Color.Black:
                    w.left.color = Color.Black
                    w.color = Color.Red
                    self._right_rotate(w)
                    w = x.p.right
                else:
                    w.color = x.p.color
                    x.p.color = Color.Black
                    w.right.color = Color.Black
                    self._left_rotate(x.p)
                    x = self.root
            else:
                w = x.p.left
                if w.color is Color.Red:
                    w.color = Color.Black
                    x.p.color = Color.Red
                    self._right_rotate(x.p)
                    w = x.p.left
                if w.right.color is Color.Black and w.left.color is Color.Black:
                    w.color = Color.Red
                    x = x.p
                elif w.left.color is Color.Black:
                    w.right.color = Color.Black
                    w.color = Color.Red
                    self._left_rotate(w)
                    w = x.p.left  # 保证一致性
                else:
                    w.color = x.p.color
                    x.p.color = Color.Black
                    w.left.color = Color.Black
                    self._right_rotate(x.p)
                    x = self.root
        x.color = Color.Black

    def _inorder_traversal(self, tn: RBTreeNode, res: List[T]) -> None:
        if tn is self.nil:
            return
        self._inorder_traversal(tn.left, res)
        res.append(tn.key)
        self._inorder_traversal(tn.right, res)

    def __repr__(self) -> str:
        """中序遍历"""
        res = []
        self._inorder_traversal(self.root, res)
        return repr(res)


class RBSortedList(Generic[T]):
    """
    Test Ref: https://leetcode.cn/problems/avoid-flood-in-the-city/
        当然只是为了测试. 如果为了速度, 可以使用sortedcontainers库.
    """

    def __init__(self, nums: Optional[List[T]] = None) -> None:
        if nums is None:
            nums = []
        self.rbt = RBTree()
        for x in nums:
            self.add(x)

    def add(self, x: T) -> None:
        self.rbt.insert(RBTreeNode(x))

    def remove(self, x: T) -> None:
        tn = self.rbt.search(x)
        assert tn is not self.rbt.nil
        self.rbt.delete(tn)

    def bisect_left(self, x: T) -> int:
        return self.rbt.bisect_left(x)

    def bisect_right(self, x: T) -> int:
        return self.rbt.bisect_right(x)

    def pop(self, i: int) -> T:
        tn = self.rbt.getitem(i)
        res = tn.key
        self.rbt.delete(tn)
        return res

    def __len__(self) -> int:
        return self.rbt.root.size

    def __getitem__(self, i: int) -> T:
        if i < 0:
            i = i + len(self)
        if i >= len(self):
            raise IndexError
        return self.rbt.getitem(i).key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.rbt!r})"


if __name__ == '__main__':
    x = RBSortedList([2, 3, 4, 6])  #
    print(x)
    print(len(x))
    print(x[1])
