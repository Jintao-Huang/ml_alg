
from libs import *
from libs.alg import *


class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        return kmp(haystack, needle)


class Solution2:
    def strStr(self, haystack: str, needle: str) -> List[int]:
        res = kmp2(haystack, needle)
        return res
    

if __name__ == "__main__":
    haystack = "sadbutsad"
    needle = "sad"
    print(Solution().strStr(haystack, needle))
    print(Solution2().strStr(haystack, needle))

    haystack = "leetcode"
    needle = "leeto"
    print(Solution().strStr(haystack, needle))
    print(Solution2().strStr(haystack, needle))