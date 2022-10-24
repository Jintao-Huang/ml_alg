from libs import *
from libs.alg import *


class Solution:
    def is_ok(self, piles: List[int], k: int, h: int) -> bool:
        res_h = 0
        for p in piles:
            res_h += math.ceil(p / k)
            if res_h > h:
                return False
        return True


    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        return bs(1, max(piles), lambda i: self.is_ok(piles, i, h))

if __name__ == "__main__":
    piles = [3,6,7,11]
    h = 8
    print(Solution().minEatingSpeed(piles, h))