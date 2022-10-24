
from typing import List, Optional, Union

__all__ = ["kmp", "kmp2", "is_palindromic"]

"""KMP
Ref: https://www.bilibili.com/video/BV1d54y1q7ko
    https://www.bilibili.com/video/BV1AY4y157yL
s: ababababaabab
sub_s: ababaab
    (最长的真前缀和真后缀相同的长度.)
    next: -1,0("a"),0,1,2,3,1("ababaa"). (idx从0开始)
        规律: 前两位一定是-1,0; 不看当前char, 只看当前字符前面的字符串.
    nextval: -1,0,-1,0,-1,3,0
        规律: 看当前字符串. 若调到前面后, 当前字符相等, 则继续往前跳.
note: 考研中的kmp算法只能匹配一次. 
    可以通过改进kmp算法, 使其能够多次匹配(见算法导论, 改造next数组的定义: 左移一位)
    令next: 0("a"),0,1,2,3,1,2("ababaab").
        这样当匹配结束后, 则跳到2这个位置.
        (看当前char的最长的真前缀和真后缀相同的长度) 
    nextval: 0,0,0,0,3,0,2
        (看下一个char, 与跳过去的字符是否相等) 
下面会采用算法导论中的定义.
"""


def _build_nextval(sub_s: str) -> List[int]:
    """获取next数组
    -: 递推方式求解(利用已经求解的next数组)
    """
    n = len(sub_s)

    if n == 0:
        return []
    nextval = [0]
    i = 0
    #
    for j in range(1, n):
        # next
        while i > 0 and sub_s[i] != sub_s[j]:
            i = nextval[i - 1]
        if sub_s[i] == sub_s[j]:
            i += 1
        # nextval
        prev_nv = nextval[-1]
        if sub_s[j] == sub_s[prev_nv]:
            nextval[-1] = nextval[prev_nv - 1]
        #
        nextval.append(i)

    return nextval


def kmp(s: str, sub_s: str, nextval: Optional[List[int]] = None) -> int:
    """求sub_s在s中匹配的开始位置
    Test Ref: https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/
    -: 如果aa, ab第二个不匹配, 则查找表格. 
    """
    assert len(sub_s) != 0
    if nextval is None:
        nextval = _build_nextval(sub_s)
    j = 0  # 对应sub_s
    res = -1
    for i in range(len(s)):  # 对应s
        # 仿_build_nextval
        while j > 0 and s[i] != sub_s[j]:
            j = nextval[j - 1]
        if s[i] == sub_s[j]:
            j += 1
        #
        if j == len(sub_s):
            return i - j + 1
    return res


def kmp2(s: str, sub_s: str, nextval: Optional[List[int]] = None) -> List[int]:
    """可以多次匹配"""
    assert len(sub_s) != 0
    if nextval is None:
        nextval = _build_nextval(sub_s)
    j = 0  # 对应sub_s
    res = []
    for i in range(len(s)):  # 对应s
        # 仿_build_nextval
        while j > 0 and s[i] != sub_s[j]:
            j = nextval[j - 1]
        if s[i] == sub_s[j]:
            j += 1
        #
        if j == len(sub_s):
            res.append(i - j + 1)
            j = nextval[j - 1]
    return res


if __name__ == "__main__":
    print(_build_nextval("aabaab"))
    print(kmp("aabaaabaab", "aabaab"))
    print(kmp2("aabaaabaab", "aabaab"))


def is_palindromic(s: Union[str, List[int]]) -> bool:
    lo, hi = 0, len(s) - 1
    while lo < hi:
        if s[lo] != s[hi]:
            return False
        lo += 1
        hi -= 1
    return True


if __name__ == "__main__":
    print(is_palindromic("12321"))
    print(is_palindromic("123321"))
    print(is_palindromic("1211"))
