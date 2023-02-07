from libs import *


class Solution:
    def minDistance(self, s1: str, s2: str) -> int:
        return libs_alg.edit_distance(s1, s2)


if __name__ == "__main__":
    word1 = "horse"
    word2 = "ros"
    print(Solution().minDistance("horse", "ros"))
