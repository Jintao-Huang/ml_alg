import sys
import os
from typing import List, Optional


IGNORE_DIRS = [".vscode", "runs", "__pycache__", ".git", "build", "mini_lightning.egg-info", "dist", "asset", ".pytest_cache"]

if __name__ == "__main__":
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *


# if __name__ == "__main__":
#     for item in mywalk("/home/jintao/Desktop/coding/python/mini-lightning"):
#         print(item)


def is_chinese(c: str) -> bool:
    if "\u4e00" <= c <= "\u9fff":
        return True
    else:
        return False


def contain_chinese(s: str) -> bool:
    for c in s:
        if is_chinese(c):
            return True
    return False


def detect_chinese(dir_: str, ignore_dirs: Optional[List[str]] = None):
    # 遍历某个文件夹中的所有文件, 若某文件的某行存在中文, 则显示文件名和行数
    iter_ = libs_utils.mywalk(dir_, ignore_dirs)
    for _, curr_dir, _, fname_list in iter_:
        for fname in fname_list:
            path = os.path.join(curr_dir, fname)
            with open(path, "r") as f:
                try:
                    for i, line in enumerate(f):
                        if contain_chinese(line):
                            print(path, i + 1)
                except UnicodeDecodeError:
                    print(f"path, {path}")


# if __name__ == "__main__":
#     s = "abc我123"
#     print(contain_chinese(s))
#     print(contain_chinese("abc123"))
#     for i in range(len(s)):
#         print(s[i].encode("utf8"), is_chinese(s[i]))


if __name__ == "__main__":
    detect_chinese("/home/jintao/Desktop/coding/python/ustcml/mini-lightning/", IGNORE_DIRS)
