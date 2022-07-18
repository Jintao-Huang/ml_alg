
import time
from typing import Callable, Any, Optional, List
import os
from urllib.parse import urljoin
from urllib.error import HTTPError
from urllib.request import urlretrieve
import numpy as np

__all__ = ["test_time", "download_files"]


def test_time(func: Callable[[], Any], number: int = 100, warm_up: int = 2, timer: Optional[Callable[[], float]] = None) -> Any:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    # 预热
    for _ in range(warm_up):
        func()
    #
    res = None
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    # 打印平均, 标准差, 最大, 最小
    ts = np.array(ts)
    max_ = ts.max()
    min_ = ts.min()
    mean = ts.mean()
    std = ts.std()
    # print
    print(
        f"time[number={number}]: {mean:.6f}±{std:.6f} |max: {max_:.6f} |min: {min_:.6f}")
    return res


def download_files(base_url: str, fnames: List[str], save_dir:str):
    os.makedirs(save_dir, exist_ok=True)
    for fname in fnames:
        if '/' in fname:
            dir = os.path.join(save_dir, os.path.dirname(fname))
            os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        if os.path.exists(save_path):
            continue
        file_url = urljoin(base_url, fname)
        print(f"Downloading `{file_url}`")
        try:
            urlretrieve(file_url, save_path)
        except HTTPError:
            raise

# if __name__ == "__main__":
#     from timeit import timeit

#     def func(x, y):
#         return x @ y
#     x = np.random.randn(1000, 1000)
#     test_time(lambda: func(x, x), 100)

#     #
#     import sys
#     sys.path.append("/home/jintao/Desktop/coding/python/ml_alg")
#     from libs import libs_ml
#     test_time(lambda: func(x, x), 100, timer=libs_ml.time_synchronize)
#     print(timeit(lambda: func(x, x), number=100))
