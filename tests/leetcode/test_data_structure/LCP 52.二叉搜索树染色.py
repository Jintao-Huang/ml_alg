from libs import *
from libs.alg import *


class Solution:
    def getNumber(self, root: Optional[TreeNode], ops: List[List[int]]) -> int:
        """
        - 先将树中序遍历, 获得list. 然后使用lazy_segment_tree进行更新
        """
        li = inorder_traversal(root)
        mapper: Dict[int, int] = {x: i for i, x in enumerate(li)}  # val -> idx
        st = LazySegmentTree(len(li), False)
        for type_, x, y in ops:
            st.update(mapper[x], mapper[y], type_)
        st.sum_range(0, len(li) - 1)
        return st.sum_range(0, len(li) - 1)


if __name__ == "__main__":
    root = to_tree("[1,null,2,null,3,null,4,null,5]")
    ops = [[1, 2, 4], [1, 1, 3], [0, 3, 5]]
    print(Solution().getNumber(root, ops))
    # 
    root = to_tree("[4,2,7,1,null,5,null,null,null,null,6]")
    ops = [[0,2,2],[1,1,5],[0,4,5],[1,5,7]]
    print(Solution().getNumber(root, ops))
    
