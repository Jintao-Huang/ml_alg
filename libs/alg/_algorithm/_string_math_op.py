
from typing import Dict, List, Tuple


__all__ = ["string_add", "string_mul", "string_to_int"]


def _string_add(x1: List[int], x2: List[int]) -> List[int]:
    """
    -: 从后往前遍历, 并设置进位符carry. 直到carry为0, 且a, b越界.
    """
    if len(x1) == 0 and len(x2) == 0:
        return [0]
    #
    i, j = len(x1) - 1, len(x2) - 1
    carry = 0
    res: List[int] = []
    while i >= 0 or j >= 0 or carry > 0:
        a = x1[i] if i >= 0 else 0
        b = x2[j] if j >= 0 else 0
        carry, r = divmod(a+b+carry, 10)
        res.append(r)
        i -= 1
        j -= 1
    # 去0
    while len(res) > 1 and res[-1] == 0:
        res.pop()
    return res[::-1]


def string_add(x1: str, x2: str) -> str:
    """
    Test Ref: https://leetcode.cn/problems/add-strings/
    """
    res = _string_add([int(a) for a in x1], [int(a) for a in x2])
    return "".join([str(a) for a in res])


if __name__ == "__main__":
    num1 = ""
    num2 = "00000"
    print(string_add(num1, num2))


def _string_mul(x1: List[int], x2: List[int]) -> List[int]:
    """
    -: 使用加法. 遍历x1为x1[i], 然后使用每一个x1[i]乘以x2得r. 然后将值r相加得res. 
        使用a乘以x2时, 将a与每一个x2的位从后往前相乘, 并将每个位数存入r_l中. 最后翻转r_l获得r. 
    """
    i = 0
    n, m = len(x1), len(x2)
    mapper: Dict[int, List[int]] = {0: [0]}  # a * x2 -> r. 因为a一共只有10个, 所以可以进行memory.
    res = [0]
    while i < n:
        a = x1[i]
        if a in mapper:
            r = mapper[a]
        else:
            r_l: List[int] = []
            j = m - 1
            carry = 0
            while j >= 0 or carry > 0:
                b = x2[j] if j >= 0 else 0
                carry, _r = divmod(a * b + carry, 10)
                r_l.append(_r)
                j -= 1
            r = r_l[::-1]
            mapper[a] = r
        res.append(0)  # *10
        res = _string_add(res, r)
        i += 1
    return res


def string_mul(x1: str, x2: str) -> str:
    """
    Test Ref: https://leetcode.cn/problems/multiply-strings/
    """
    res = _string_mul([int(a) for a in x1], [int(a) for a in x2])
    return "".join([str(a) for a in res])


if __name__ == "__main__":
    num1 = "0"
    num2 = "0"
    print(string_mul(num1, num2))


def string_to_int(s: str) -> int:
    """
    Test Ref: https://leetcode.cn/problems/string-to-integer-atoi/
    """
    n = len(s)
    res = 0
    ord_0 = ord('0')
    for i in range(n):
        res *= 10
        res += ord(s[i]) - ord_0
    return res


if __name__ == "__main__":
    s = "42"
    print(string_to_int(s))
