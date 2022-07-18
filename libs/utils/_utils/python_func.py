def _partial(func, *args, **kwargs):
    def new_func(*args2, **kwargs2):
        return func(*args, *args2, **kwargs, **kwargs2)
    return new_func


# if __name__ == "__main__":
#     def func(x, y, z):
#         return x * y * z

#     from functools import partial
#     z = partial(func, 1, z=2)
#     z2 = _partial(func, 1, z=2)
#     z3 = lambda y: func(1, y, 2)
#     zis = [lambda y: func(1, y, z) for z in range(10)]
#     print(z(1))
#     print(z2(1))
#     print(z3(1))
#     for zi in zis:
#         print(zi(1))
