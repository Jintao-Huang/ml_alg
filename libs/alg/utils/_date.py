
from typing import Tuple
"""
date: 20110102. 2011年1月2日.
"""

__all__ = ["get_ymd", "is_leap_year", "calc_date_day", "calc_date_diff"]


def get_ymd(date: int) -> Tuple[int, int, int]:
    return date // 10000, date % 10000 // 100, date % 100


def is_leap_year(year: int) -> bool:
    if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
        return True
    return False


def _n_leap_year(year: int) -> int:
    """从1年开始的闰年数量. 不含year."""
    year -= 1
    n = year // 4
    n -= year // 100
    n += year // 400
    return n


if __name__ == "__main__":
    print(_n_leap_year(2020))
    print(_n_leap_year(2019))
    #
    print(_n_leap_year(2000))
    print(_n_leap_year(1999))
    #
    print(_n_leap_year(100))
    print(_n_leap_year(99))
    print()
    """
490
489
485
484
24
24
"""


def calc_date_day(date: int) -> int:
    """计算是一年中的第几天. 0101为第1天."""
    m_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    y, m, d = get_ymd(date)
    ily = is_leap_year(y)
    y_d = m_days[m - 1] + d
    if ily and m >= 3:
        y_d += 1
    return y_d


if __name__ == "__main__":
    print(calc_date_day(20200101))
    print(calc_date_day(20200301))
    print(calc_date_day(20201231))
    print(calc_date_day(20190301))
    print()
    """
1
61
366
60
"""


def calc_date_diff(date1: int, date2: int) -> int:
    """date1 - date2的天数之差.
    -: 先计算date1, date2距离1月1日的天数, y_d1, y_d2
        然后将date1的y1 - date2的y2.
    复杂度: O(1)
    """
    y_d1 = calc_date_day(date1)
    y_d2 = calc_date_day(date2)
    y1, _, _ = get_ymd(date1)
    y2, _, _ = get_ymd(date2)
    d = y_d1 - y_d2
    d += (y1 - y2) * 365
    d += (_n_leap_year(y1) - _n_leap_year(y2))
    return d


if __name__ == "__main__":
    print(calc_date_diff(20191231, 20200101))
    print(calc_date_diff(20200101, 20190101))
    print(calc_date_diff(20201231, 20191231))
    """
-1
365
366
"""


def _check_date_gt(date1: int, date2: int) ->bool:
    """date1 > date2?"""
    return date1 > date2

if __name__ == "__main__":
    print()
    print(_check_date_gt(20200101, 20191231))  # True
    print(_check_date_gt(20200301, 20200229))  # True