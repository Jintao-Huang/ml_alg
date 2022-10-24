from libs import *
from libs.alg import *


class Solution:
    def sumScores(self, s: str) -> int:
        """会有一定概率出错. 因为hash碰撞的存在. 所以该算法是不完备的. 这里只是为了测试StringHasher"""
        n = len(s)
        sh = StringHasher(s)
        res = 0
        for i in range(1, n + 1):  # i表示长度.
            # lo..hi
            if s[0] != s[n-i]:
                continue
            idx = bs2(1, i, lambda j: sh.get_hash(0, j) == sh.get_hash(n - i, n-i+j))
            res += idx
        return res


if __name__ == "__main__":
    s = "wozglgylxobrmlutkqyfmoihxenvdrpscksauivgpkfcgevsznwwozglgy"
    print(len(s))
    print(Solution().sumScores(s))
