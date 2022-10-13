"""test date"""
from libs.alg import *


def get_date(date: str) -> int:
    y, m, d = date.split("-")
    return int(y) * 10000 + int(m) * 100 + int(d)


class Solution:
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        d1: int = get_date(date1)
        d2: int = get_date(date2)
        return abs(calc_date_diff(d1, d2))
