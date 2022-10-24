
from typing import List, Dict, Any
from functools import partial

__all__ = ["call_callable_list"]


def call_callable_list(
        callable_list: List[str],
        args_list: List[List[Any]],
        globals: Dict[str, Any]
) -> List[Any]:
    """调用一系列可调用的函数或类. 返回可调用类/函数的返回.
    思路: 循环callable_list, args_list. 获取callable_str, args. 并从globals获取callable_obj. 并获取res
    """
    res = []
    g = globals.copy()
    for callable_str, args in zip(callable_list, args_list):
        callable_obj = g[callable_str]
        r = callable_obj(*args)
        if isinstance(callable_obj, type):
            g.update({k: partial(v, r) for k, v in callable_obj.__dict__.items() if callable(v)})
        res.append(r)
    return res


if __name__ == "__main__":
    li = [1, None, 2, 3, None, 4, 5]
    tree = call_callable_list(["to_tree"], [[li]], globals())[0]
    li2 = call_callable_list(["from_tree"], [[tree]], globals())[0]
    print(li2)

if __name__ == "__main__":
    class A:
        def aaa(self):
            print(1)

    a = A()
    a.aaa()
    print()
    print(call_callable_list(["A", "aaa"], [[], []], globals()))
