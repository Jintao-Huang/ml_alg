
from libs import *
from libs.alg import *

class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        p = ListNode(0, head)
        head = p
        for _ in range(left - 1):
            p = p.next
        pv = p
        p = p.next
        p2 = p
        for _ in range(right - left + 1):
            p2 = p2.next
        p3 = reverse_list(p, p2)
        pv.next = p3
        return head.next

if __name__ == "__main__":
    head = to_list([1, 2, 3, 4, 5])
    left = 2
    right = 4
    print(from_list(Solution().reverseBetween(head, left, right)))
        
