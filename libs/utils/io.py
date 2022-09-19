# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import os
import pickle
from typing import Any
import json
import yaml
import pandas as pd
from pandas import DataFrame

__all__ = [
    "read_from_pickle", "save_to_pickle",
    "read_from_json", "save_to_json",
    "read_from_yaml", "save_to_yaml",
    "read_from_csv", "save_to_csv"
]


def read_from_pickle(fpath: str,) -> Any:
    with open(fpath, "rb") as f:
        res = pickle.load(f)
    return res


def save_to_pickle(obj: Any, fpath: str) -> None:
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)


def read_from_json(fpath: str, encoding: str = "utf-8") -> Any:
    with open(fpath, "r", encoding=encoding) as f:
        res = json.load(f)
    return res


def save_to_json(obj: Any, fpath: str, encoding: str = "utf-8") -> None:
    with open(fpath, "w", encoding=encoding) as f:
        json.dump(obj, f)


def read_from_yaml(fpath: str, encoding: str = "utf-8", loader=None) -> Any:
    loader = yaml.SafeLoader if loader is None else loader
    with open(fpath, "r", encoding=encoding) as f:
        res = yaml.load(f, loader)
    return res


def save_to_yaml(obj: Any, fpath: str, encoding: str = "utf-8", mode: str = "w") -> None:
    with open(fpath, mode, encoding=encoding) as f:
        yaml.dump(obj, f)


def read_from_csv(fpath: str, *, sep: str = ",") -> DataFrame:
    return pd.read_csv(fpath, sep=sep)


def save_to_csv(df: DataFrame, fpath: str, *, sep: str = ",", index: bool = False) -> None:
    df.to_csv(fpath, sep=sep, index=index)


# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.pkl"
#     obj = ({"123"}, [123, ], 1.1)
#     save_to_pickle(obj, fpath)
#     obj = read_from_pickle(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.json"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     save_to_json(obj, fpath)
#     obj = read_from_json(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.yaml"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     save_to_yaml(obj, fpath)
#     obj = read_from_yaml(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     FPAHT1 = "/home/jintao/Desktop/coding/python/private/asset/updated_fileindex.csv"
#     FPATH2 = "/home/jintao/Desktop/coding/python/private/asset/monthly_return.csv"
#     FPATH3 = "/home/jintao/Desktop/coding/python/private/asset/cik_ticker.csv"
#     FPATH4= "./asset/1.csv"
#     import mini_lightning as ml
#     ml.test_time(lambda: read_from_csv(FPAHT1))
#     df = ml.test_time(lambda: read_from_csv(FPATH3, sep="|"))
#     save_to_csv(df, FPATH4)
#     os.remove(FPATH4)
