# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from .._types import *
from mini_lightning import write_to_yaml, read_from_yaml, write_to_csv

def read_from_file(fpath: str, mode: str = "r") -> Union[str, bytes]:
    with open(fpath, mode, encoding="utf-8") as f:
        text = f.read()
    return text


def write_to_file(text: Union[str, bytes], fpath: str, mode: str = "w") -> None:
    with open(fpath, mode, encoding="utf-8") as f:
        f.write(text)


# def torch_load(fpath: str, map_location: Optional[Device]) -> Any:
#     return torch.load(fpath, map_location)


# def torch_save(obj: Any, fpath: str) -> None:
#     torch.save(obj, fpath)


def read_from_pickle(fpath: str) -> Any:
    with open(fpath, "rb") as f:
        res = pickle.load(f)
    return res


def write_to_pickle(obj: Any, fpath: str) -> None:
    with open(fpath, "wb") as f:
        pickle.dump(obj, f)


def read_from_json(fpath: str, encoding: str = "utf-8") -> Any:
    with open(fpath, "r", encoding=encoding) as f:
        res = json.load(f)
    return res


def write_to_json(obj: Any, fpath: str, encoding: str = "utf-8") -> None:
    with open(fpath, "w", encoding=encoding) as f:
        json.dump(obj, f)


def read_from_csv_df(fpath: str, *, sep: str = ",", **kwargs) -> DataFrame:
    return pd.read_csv(fpath, sep=sep, **kwargs)


def write_to_csv_df(df: DataFrame, fpath: str, *, sep: str = ",", index: bool = False) -> None:
    df.to_csv(fpath, sep=sep, index=index)


def read_from_csv(fpath: str, nrows: int = -1, *, sep: str = ",", encoding="utf-8") -> List[List[str]]:
    res = []
    n = 0
    if nrows == 0:
        return res
    #
    with open(fpath, "r", newline="", encoding=encoding) as f:
        reader = csv.reader(f, delimiter=sep)
        for l in reader:
            res.append(l)
            n += 1
            if nrows > 0 and nrows == n:
                break
    return res

# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.pkl"
#     obj = ({"123"}, [123, ], 1.1)
#     write_to_pickle(obj, fpath)
#     obj = read_from_pickle(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.json"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     write_to_json(obj, fpath)
#     obj = read_from_json(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     import os
#     fpath = "asset/1.yaml"
#     obj = [{"123": "aaa"}, [123, ], 1.1]
#     write_to_yaml(obj, fpath)
#     obj = read_from_yaml(fpath)
#     print(obj)
#     os.remove(fpath)


# if __name__ == "__main__":
#     FPAHT1 = "/home/jintao/Desktop/0_coding/0_python/2_ICS/ics/.dataset/stock/monthly_return.csv"
#     FPATH3 = "/home/jintao/Desktop/0_coding/0_python/2_ICS/ics/.dataset/cik_ticker/1.csv"
#     FPATH4 = "./asset/1.csv"
#     FPATH5 = "./asset/2.csv"
#     import mini_lightning as ml
#     ml.test_time(lambda: read_from_csv(FPAHT1))
#     obj = ml.test_time(lambda: read_from_csv(FPATH3, sep="|"))
#     ml.test_time(lambda: write_to_csv(obj, FPATH4))
#     os.remove(FPATH4)
#     #
#     ml.test_time(lambda: read_from_csv_df(FPAHT1))
#     df = ml.test_time(lambda: read_from_csv_df(FPATH3, sep="|"))
#     ml.test_time(lambda: write_to_csv_df(df, FPATH4))
#     os.remove(FPATH4)

def read_from_npy(fpath: str) -> ndarray:
    return np.load(fpath)

def write_to_npy(array: ndarray, fpath: str) -> None:
    return np.save(fpath, array)