from ._search_nb import *
from ._sort_nb import *
from ._sort_cy import quick_sort_cy as _quick_sort_cy, merge_sort_cy as _merge_sort_cy
from numpy import ndarray


def quick_sort_cy(nums: ndarray) -> None:
    _quick_sort_cy(nums)


def merge_sort_cy(nums: ndarray) -> None:
    _merge_sort_cy(nums)
