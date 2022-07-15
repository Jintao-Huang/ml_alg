
import time
from timeit import timeit
from typing import Callable, Any
import numpy as np


def test_time(func: Callable[[], Any], number: int = 100, warm_up: int = 5, timer: Callable[[], float] = None) -> Any:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    # 预热
    res = func()
    for _ in range(warm_up - 1):
        func()
    #
    for _ in range(number):
        t1 = timer()
        func()
        t2 = timer()
        ts.append(t2 - t1)
    # 打印平均, 标准差, 最大, 最小
    ts = np.array(ts)
    n = ts.shape[0]
    max_ = ts.max()
    min_ = ts.min()
    mean = ts.mean()
    std = ts.std()
    # print
    print(f"time: {mean:.6f}±{std:.6f} |max: {max_:.6f} |min: {min_:.6f}")
    return res


if __name__ == "__main__":
    def func(x, y):
        return x @ y
    x = np.random.randn(1000, 1000)
    test_time(lambda: func(x, x), 100)

    #
    import sys
    sys.path.append("/home/jintao/Desktop/coding/python/ml/")
    from libs import libs_ml
    test_time(lambda: func(x, x), 100, timer=libs_ml.time_synchronize)
    print(timeit(lambda: func(x, x), number=100))
    #
