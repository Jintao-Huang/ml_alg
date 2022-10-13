
from libs import *
from libs.alg import *


class Solution:
    def fraction(self, cont: List[int]) -> List[int]:
        """使用链表的后序遍历
        3 + 1/(2+1/(0+1/(2)))"""
        n = len(cont)
        res0, res1 = cont[-1], 1
        for i in reversed(range(n - 1)):
            x = cont[i]
            res0, res1 = self.fraction_reduction(x * res0 + 1 * res1,  res0)
        return [res0, res1]

    @staticmethod
    def fraction_reduction(x, y) -> Tuple[int, int]:
        """约分"""
        _gcd = gcd(x, y)
        return x // _gcd, y // _gcd


if __name__ == "__main__":
    cont = [3, 2, 0, 2]
    print(Solution().fraction(cont))
    # [13, 4]
