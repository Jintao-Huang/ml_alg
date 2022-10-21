try:
    from .._algorithm._utils import accumulate
except ImportError:
    from libs.alg import accumulate

__all__ = ["StringHasher"]


class StringHasher:
    """
    -: 传入字符串. 获取hash时, 传入lo, hi. 获取s[lo:hi](不含hi)的hash. 时间复杂度O(1)
        存储前缀和. 利用前缀和之差达到O(1)复杂度.
        e.g. "abc" = a*27**2 + b*27 + c; "c" = hash("abc") - hash("ab")*27
    Test Ref: https://leetcode.cn/problems/sum-of-scores-of-built-strings/
    """

    def __init__(self, s: str,
                 min_char: str = "a", max_char: str = "z", mod: int = int(1e9)+7) -> None:
        """[min_char..max_char]"""
        n = len(s)
        self.s = s
        self.min_char = ord(min_char)  # 代表`1`
        self.max_char = ord(max_char)
        base_o = self.max_char - self.min_char + 2
        self.mod = mod
        #
        self.prefix_sum = accumulate(s, lambda x, y: (x * base_o + (ord(y) - self.min_char + 1) % mod), [0])
        self.base = accumulate(range(n - 1), lambda b, _: ((b * base_o) % mod), [1])

    def get_hash(self, lo: int, hi: int) -> int:
        """[lo,hi)"""
        assert lo <= hi
        if lo == hi:  # 空字符串
            return 0
        if lo == 0:
            return self.prefix_sum[hi]
        return (self.prefix_sum[hi] - self.prefix_sum[lo] * self.base[hi - lo]) % self.mod


if __name__ == "__main__":
    sh = StringHasher("abc")
    print(sh.prefix_sum, sh.base)
    print(sh.get_hash(2, 3))
    print(sh.get_hash(1, 3))  # 2*27+3
