from libs import *

if __name__ == "__main__":
    # 快 < ndarray.sort < list.sort < quick_sort(x） < quick_sort(x_tl)
    #   < quick_sort(x_l) < python_quick_sort(x_l) < 慢
    x = np.random.randn(1000000)
    x_l = libs_ml.test_time(lambda: x.tolist())
    libs_ml.test_time(lambda: np.sort(x), 1)  # fast
    libs_ml.test_time(lambda: x_l.sort(), 1)
    libs_ml.test_time(lambda: libs_algf.partition_cy(x.copy()))  # fast
    libs_ml.test_time(lambda: libs_algf.quick_sort_cy(x.copy()), 1, 1)  # fast
    libs_ml.test_time(lambda: libs_algf.merge_sort_cy(x.copy()), 1, 1)  # fast

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5, 6, 7, 2, 4, 5, 7, 8.])
    libs_algf.merge_cy(x, 0, 6, x.shape[0] - 1)
    print(x)
