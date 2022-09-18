
from typing import Dict


def dict_sorted_key(d: Dict[int, int]) -> Dict[int, int]:
    """不通过sort value进行输出, 而是通过sort key"""
    keys = sorted(d.keys())
    res = {}
    for k in keys:
        res[k] = d[k]
    return res


if __name__ == "__main__":
    print(dict_sorted_key({1: 2, 3: 4, 5: 3}))
