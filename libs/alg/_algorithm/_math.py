from math import gcd, lcm, factorial, comb, perm
from math import isnan, isinf, inf, nan, e, pi
import math
from typing import Optional, Union, overload, Literal, List, Dict
from collections import Counter


__all__ = ["gcd", "lcm", "factorial", "comb", "perm",
           "pow2_int",
           "isnan", "isinf", "inf", "nan", "e", "pi",
           "is_prime_num", "find_prime_nums"]


def _gcd(x: int,  y: int) -> int:
    """辗转相除法: Ref: https://zh.m.wikipedia.org/zh-hans/%E8%BC%BE%E8%BD%89%E7%9B%B8%E9%99%A4%E6%B3%95#%E8%AE%A1%E7%AE%97%E6%9C%BA%E5%AE%9E%E7%8E%B0
    思路: 使用辗转相除法. 使用x mod y. 直到y==0. 
    Test Ref: 分子分母约分: https://leetcode.cn/problems/deep-dark-fraction
    """
    while y > 0:
        x, y = y, x % y
    return x


def _lcm(x: int, y: int) -> int:
    """
    思路: 使用gcd进行计算. 
    """
    return x * y // _gcd(x, y)


if __name__ == "__main__":
    x = 100
    y = 128
    print(_gcd(x, y), gcd(x, y))
    print(_lcm(x, y), lcm(x, y))
    print(gcd(x, y, 244))


def _comb(n: int, k: int) -> int:
    """n个中取k个的组合数
    思路: n! / k! / (n-k)! 这里对时间复杂度不进行优化. 
    """
    return _perm(n, k) // _factorial(k)


def _perm(n: int, k: int) -> int:
    """n个中取k个的排列数
    思路: n! / (n-k)!. 这里对时间复杂度不进行优化. 
    """
    return _factorial(n) // _factorial(n-k)


def _factorial(x: int) -> int:
    """x!"""
    res = 1
    for i in range(2, x + 1):
        res *= i
    return res


if __name__ == "__main__":
    print(_comb(15, 5), comb(15, 5))
    print(_perm(14, 10), perm(14, 10))
    print(_factorial(10), factorial(10))


def pow2_int(y: int) -> int:
    return 1 << y


if __name__ == "__main__":
    import mini_lightning as ml
    y = ml.test_time(lambda: 2 ** 10000, 10)
    y2 = ml.test_time(lambda: pow2_int(10000), 10)
    print(y == y2)


def _python_mod(x: int, y: int) -> int:
    """
    原理: 等价于c++中的 (x % y + y) % y. 不返回负数. 
    """
    raise NotImplementedError


@overload
def _fast_pow(x: float, y: int, mod: Union[float, int, None] = None) -> float: ...
@overload
def _fast_pow(x: int, y: int, mod: Optional[int] = None) -> int: ...


def _fast_pow(x, y, mod=None):
    """
    思路: 如果y为偶数: x ** y == (x ** 2) ** (y//2)
        如果y为奇数: x ** y == x * x ** (y-1)
        使y //= 2或 y-=1, 直到y==0. x**0==1
    mod: 常为int(1e9)+7. 在c++中可以令res, x为long long型, 避免数值溢出. 
    """
    res = 1
    while y > 0:
        if y % 2 == 0:
            x *= x
            y //= 2
            if mod is not None:
                x %= mod
        else:  # == 1
            y -= 1
            res *= x
            if mod is not None:
                res %= mod
    return res


if __name__ == "__main__":
    print(ml.test_time(lambda: _fast_pow(3, 10345, int(1e9) + 7)))
    print(ml.test_time(lambda: _fast_pow(4.3, 10345, 1e9 + 7)))
    print(ml.test_time(lambda: pow(3, 10345, int(1e9) + 7)))


def is_prime_num(n: int) -> bool:
    """
    思路: 遍历从[2..int(sqrt(n))]. 若n都不能除通, 则返回True. 
    复杂度: O(sqrtn)
    """
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _eratosthenes(n: int) -> List[int]:
    """返回从[2..n]之间的所有的质数(含n). 
    eratosthenes筛法. 
        Ref: https://zh.wikipedia.org/wiki/%E5%9F%83%E6%8B%89%E6%89%98%E6%96%AF%E7%89%B9%E5%B0%BC%E7%AD%9B%E6%B3%95#Python_3.6
    思路: 设置状态数组: 是否是素数的数组.
        遍历[2..int(sqrt(n))]. 若某数i是质数. 则遍历i**2, i**2+i, i**2+2i..都设为非质数.
            为什么从i**2开始遍历: i*(i-1)在遍历i-1时已经判断过了. 以此类推. 
        遍历状态数组, 若True, 则加入res. 
    复杂度: O(n loglogn)
    """
    res = []
    if n < 2:
        return res

    # is_prime[0,1]无效.
    is_prime = [True] * (n + 1)
    for i in range(2, int(math.sqrt(n) + 1)):
        if not is_prime[i]:
            continue

        for j in range(i * i, n + 1, i):
            is_prime[j] = False
    #
    for i in range(2, n + 1):
        if is_prime[i]:
            res.append(i)
    return res


def _naive(n: int) -> List[int]:
    """复杂度: O(n sqrtn)"""
    res = []
    if n < 2:
        return res
    #
    for i in range(2, n + 1):
        if is_prime_num(i):
            res.append(i)
    return res


def find_prime_nums(n: int, algo: Literal["naive", "fast"] = "fast") -> List[int]:
    """返回从[2..n]之间的所有的质数(含n)"""
    if algo == "naive":
        return _naive(n)
    elif algo == "fast":
        return _eratosthenes(n)
    else:
        raise ValueError(f"algo: {algo}")


if __name__ == "__main__":
    print(find_prime_nums(97, "naive"))
    print(find_prime_nums(97, "naive"))
    print(find_prime_nums(96, "naive"))
    print(find_prime_nums(96, "fast"))


def decomposition_prime_factor(x: int) -> Counter:
    """分解质因数
    思路: 获取[2..x]之间的质数. 
        遍历这些质数. 然后将x循环的除以这些素数, 直到x==1.
        循环直到x为1. 
    """
    res = Counter()
    if x < 2:
        return res
    #
    prime_nums = find_prime_nums(x)
    for i in range(len(prime_nums)):
        if x == 1:
            break
        #
        prime: int = prime_nums[i]
        cnt = 0
        while x % prime == 0:
            x //= prime
            cnt += 1
        #
        if cnt > 0:
            res[prime] = cnt
    return res


if __name__ == "__main__":
    cnt = decomposition_prime_factor(12342)
    print(cnt)
    print(list(cnt.elements()))


def get_factor_count(x: int) -> int:
    """获取因数的个数. 
    思路: 
        方法1: 先获取质因数, 然后利用"组合原理"计算因数个数. 
            e.g. 质因数是: {2:3, 3:4, 5:3}. 则因数=2^a*3^b*5^c. a,b,c分别有4(0..3),5,4种选择. 
                共有4*5*4个因数.
        方法2: 遍历[1..int(sqrt(x))+1]. 若i == x//j, 则res+=1; 否则res+=2.
    """
    raise NotImplementedError
