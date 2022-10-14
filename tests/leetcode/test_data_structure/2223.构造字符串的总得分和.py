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
            # 二分法: 使用长度作为二分的依据
            # 最小长度为1, 最大长度为i
            # 找最长的长度, 使得两字符串相等
            lo = 1
            hi = i
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if sh.get_hash(0, mid) == sh.get_hash(n- i, n- i + mid):
                    lo = mid
                else:
                    hi = mid - 1
            res += lo
        return res
                    
if __name__ == "__main__":
    s = "wozglgylxobrmlutkqyfmoihxenvdrpscksauivgpkfcgevsznwwozglgy"
    print(len(s))
    print(Solution().sumScores(s))

