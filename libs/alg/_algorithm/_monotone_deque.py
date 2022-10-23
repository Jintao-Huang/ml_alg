from typing import List, Iterable, Callable, Tuple
from collections import deque
try:
    from .._utils._climit import INT32_INF
except ImportError:
    from libs.alg import INT32_INF
__all__ = [
    "next_gt", "prev_gt", "next_lt", "prev_lt", "next_ge",
    "next_gt2",
    "next_ge_prev_gt",
    "next_ge_min",
    "next_k_max", "prev_k_max", "next_k_min",
    "next_ge_k_len", "prev_le_k_len",
    "largest_rect"
]


def _1(nums: List[int], iter_range: Callable[[int], Iterable], comp: Callable[[int, int], bool]) -> List[int]:
    n = len(nums)
    res = [-1] * n
    stack = []
    for i in iter_range(n):
        x = nums[i]
        while len(stack) > 0 and comp(nums[stack[-1]],  x):
            stack.pop()
        if len(stack) > 0:
            res[i] = stack[-1]
        stack.append(i)
    return res


def _2(nums: List[int], iter_range: Callable[[int], Iterable], comp: Callable[[int, int], bool]) -> List[int]:
    n = len(nums)
    res = [-1] * n
    stack = []
    for i in iter_range(n):
        x = nums[i]
        while len(stack) > 0 and comp(nums[stack[-1]], x):
            res[stack.pop()] = i
        stack.append(i)
    # while len(stack) > 0:
    #     res[stack.pop()] = -1
    return res


def _3(nums: List[int], iter_range: Callable[[int], Iterable], comp: Callable[[int, int], bool]) -> Tuple[List[int], List[int]]:
    """同时"""
    n = len(nums)
    res, res2 = [-1] * n, [-1] * n
    stack = []
    for i in iter_range(n):
        x = nums[i]
        while len(stack) > 0 and comp(nums[stack[-1]], x):
            res2[stack.pop()] = i
        if len(stack) > 0:
            res[i] = stack[-1]
        stack.append(i)
    return res, res2


def next_gt(nums: List[int]) -> List[int]:
    """(逆向; 递减栈). 找某元素, 下一个大于当前元素的索引
    -: 使用反向循环, 使用递减栈. 加入队列时, 邻近那个元素就是next_gt
    """
    return _1(nums, lambda n: reversed(range(n)), lambda x, y: x <= y)


def next_gt2(nums: List[int]) -> List[int]:
    """(正向; 非递增栈). 找某元素, 下一个大于当前元素的索引. 
    -: 使用正向循环, 使用非递增栈. 若遇到一个比栈顶元素>=的数, 则弹出栈顶, 并将栈顶元素的next_gt为当前元素.
    """
    return _2(nums, lambda n: range(n), lambda x, y: x < y)


def prev_gt(nums: List[int]) -> List[int]:
    """(正向; 递减栈)"""
    return _1(nums, lambda n: range(n), lambda x, y: x <= y)


def next_lt(nums: List[int]) -> List[int]:
    """(逆向; 递增栈)
    Test Ref: https://leetcode.cn/problems/largest-rectangle-in-histogram/
    """
    return _1(nums, lambda n: reversed(range(n)), lambda x, y: x >= y)


def prev_lt(nums: List[int]) -> List[int]:
    """(正向; 递增栈)
    Test Ref: https://leetcode.cn/problems/largest-rectangle-in-histogram/
    """
    return _1(nums, lambda n: range(n), lambda x, y: x >= y)


def next_ge(nums: List[int]) -> List[int]:
    """(逆向; 非递增栈)"""
    return _1(nums, lambda n: reversed(range(n)), lambda x, y: x < y)


def next_ge_prev_gt(nums: List[int]) -> Tuple[List[int], List[int]]:
    """(逆向; 非递增栈). 同时
    Test Ref: https://leetcode.cn/problems/trapping-rain-water/
        当然可以分开求, 只是速度会慢一点.
    """
    return _3(nums, lambda n: reversed(range(n)), lambda x, y: x < y)


def next_ge_min(nums: List[int]) -> List[int]:
    """下一个最近最小的大于等于当前元素的索引. 
    -: 使用稳定排序idx. 然后找idx_list中下一个大于自己的索引. 
        这保证了>=, 也保证了最小值, 也保证了最近. 
    """
    n = len(nums)
    idx_list = sorted(range(n), key=lambda i: nums[i])
    res = [-1] * n
    ng = next_gt(idx_list)
    for i in range(n):
        if ng[i] == -1:
            continue
        res[idx_list[i]] = idx_list[ng[i]]
    return res


if __name__ == "__main__":
    nums = [3, 4, 2, 2, 5, 4]
    print(next_gt(nums))
    print(next_gt2(nums))
    print(prev_gt(nums))
    print(next_lt(nums))
    print(next_ge(nums))
    print(next_ge_prev_gt(nums))
    print(next_ge_min(nums))


def _4(nums: List[int], k: int, iter_range: Callable[[int], Iterable], comp: Callable[[int, int], bool]) -> List[int]:
    n = len(nums)
    dq = deque()
    res = [-1] * n
    for i in iter_range(n):
        x = nums[i]
        while len(dq) > 0 and comp(nums[dq[-1]], x):
            dq.pop()
        dq.append(i)
        while abs(dq[0] - i) >= k:
            dq.popleft()
        #
        assert len(dq) > 0
        res[i] = dq[0]
    return res


def next_k_max(nums: List[int], k: int) -> List[int]:
    """某元素的最近的后k(含)个位置的最大值的索引. 含当前位置.
    -: 反向; 递减栈(最大值在dp[0])
        栈中保存的内容只有k格内的内容. 
        先加入当前元素到递减栈. 然后抛弃栈内元素直到只含k个位置. 栈顶元素为最大值索引
    """
    return _4(nums, k, lambda n: reversed(range(n)), lambda x, y: x <= y)


def prev_k_max(nums: List[int], k: int) -> List[int]:
    """正向; 递减栈
    Test Ref: https://leetcode.cn/problems/sliding-window-maximum/
    """
    return _4(nums, k, lambda n: range(n), lambda x, y: x <= y)


def next_k_min(nums: List[int], k: int) -> List[int]:
    """正向; 递增栈"""
    return _4(nums, k, lambda n: reversed(range(n)), lambda x, y: x >= y)


if __name__ == "__main__":
    print()
    nums = [3, 4, 2, 2, 5, 4]
    print(next_k_max(nums, 2))
    print(prev_k_max(nums, 2))
    print(next_k_min(nums, 2))


def _5(
    nums: List[int], k: int,
    iter_range: Callable[[int], Iterable],
    comp: Callable[[int, int], bool]
) -> int:
    assert k > 0
    n = len(nums)
    res = INT32_INF
    dq = deque()
    for i in iter_range(n):
        x = nums[i]
        while len(dq) > 0 and comp(nums[dq[-1]], x):
            dq.pop()
        while len(dq) > 0 and abs(nums[dq[0]] - x) >= k:
            res = min(res, abs(i - dq.popleft()))
        #
        dq.append(i)
    if res == INT32_INF:
        res = -1
    return res


def next_ge_k_len(nums: List[int], k: int) -> int:
    """下一个最近的比其大k的最短距离(不含自己)(逆向; 递减栈)
    -: 若某个数-dp[0]>=k, 则该数之前的数若更小, 但与dp[0]的距离len一定更大. 
        所以若已经满足>=k, 则dp[0]可以舍弃. 
    Test Ref: https://leetcode.cn/problems/shortest-subarray-with-sum-at-least-k/
    """
    return _5(nums, k, lambda n: reversed(range(n)), lambda x, y: x <= y)


def prev_le_k_len(nums: List[int], k: int) -> int:
    """和next_ge_k_len等价"""
    return _5(nums, k, lambda n: range(n), lambda x, y: x >= y)


if __name__ == "__main__":
    print()
    nums = [1, 2, 3, 4, 5]
    print(next_ge_k_len(nums, 1))
    print(next_ge_k_len(nums, 2))
    print(next_ge_k_len(nums, 3))
    print(prev_le_k_len(nums, 1))
    print(prev_le_k_len(nums, 2))
    print(prev_le_k_len(nums, 3))


def largest_rect(nums: List[int]) -> int:
    """
    Test Ref: https://leetcode.cn/problems/largest-rectangle-in-histogram/
        https://leetcode.cn/problems/maximal-rectangle/
    """
    nlt = next_lt(nums)
    plt = prev_lt(nums)
    n = len(nums)
    res = 0
    for i in range(n):
        j = nlt[i]
        k = plt[i]
        if j == -1:
            j = n
        if k == -1:
            k = -1
        w = j - k - 1
        res = max(res, nums[i] * w)
    return res


if __name__ == "__main__":
    heights = [2, 1, 2]
    print(largest_rect(heights))
