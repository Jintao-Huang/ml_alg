from libs import *
x = np.random.randn(100000)
libs_ml.test_time(lambda: libs_algf.quick_sort_cy(x))
