
from libs import *
from libs.alg import *


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        init_value = INT32_INF
        res = kC_full_min(coins, amount, init_value=init_value)
        return res if res != init_value else -1


class Solution2:
    def coinChange(self, coins: List[int], amount: int) -> int:
        init_value = INT32_INF
        res = kC_full_min2(coins, amount, init_value=init_value)
        return res if res != init_value else -1


if __name__ == "__main__":
    coins = [1, 2, 5]
    amount = 11
    print(Solution().coinChange(coins, amount))
    print(Solution2().coinChange(coins, amount))
