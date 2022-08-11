# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import pickle
from typing import Any

__all__ = ["read_from_pickle", "save_to_pickle"]


def read_from_pickle(file_path: str) -> Any:
    with open(file_path, "rb") as f:
        res = pickle.load(f)
    return res


def save_to_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

# if __name__ == "__main__":
#     import os
#     file_path = "./1.pkl"
#     obj = ({"123"}, [123, ], 1.1)
#     save_to_pickle(obj, file_path)
#     obj = read_from_pickle(file_path)
#     print(obj)
#     os.remove(file_path)
