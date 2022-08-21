# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import pickle
from typing import Any
import json
import yaml

__all__ = ["read_from_pickle", "save_to_pickle", "read_from_json", "save_to_json", "read_from_yaml", "save_to_yaml"]


def read_from_pickle(file_path: str,) -> Any:
    with open(file_path, "rb") as f:
        res = pickle.load(f)
    return res


def save_to_pickle(obj: Any, file_path: str) -> None:
    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_from_json(file_path: str, encoding: str = "utf-8") -> Any:
    with open(file_path, "r", encoding=encoding) as f:
        res = json.load(f)
    return res


def save_to_json(obj: Any, file_path: str, encoding: str = "utf-8") -> None:
    with open(file_path, "w", encoding=encoding) as f:
        json.dump(obj, f)


def read_from_yaml(file_path: str, encoding: str = "utf-8", loader=None) -> Any:
    loader = yaml.SafeLoader if loader is None else loader
    with open(file_path, "r", encoding=encoding) as f:
        res = yaml.load(f, loader)
    return res


def save_to_yaml(obj: Any, file_path: str, encoding: str = "utf-8", mode: str = "w") -> None:
    with open(file_path, mode, encoding=encoding) as f:
        yaml.dump(obj, f)

# if __name__ == "__main__":
#     import os
#     file_path = "asset/1.pkl"
#     obj = ({"123"}, [123, ], 1.1)
#     save_to_pickle(obj, file_path)
#     obj = read_from_pickle(file_path)
#     print(obj)
#     os.remove(file_path)


# if __name__ == "__main__":
#     import os
#     file_path = "asset/1.json"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     save_to_json(obj, file_path)
#     obj = read_from_json(file_path)
#     print(obj)
#     os.remove(file_path)


# if __name__ == "__main__":
#     import os
#     file_path = "asset/1.yaml"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     save_to_yaml(obj, file_path)
#     obj = read_from_yaml(file_path)
#     print(obj)
#     os.remove(file_path)
