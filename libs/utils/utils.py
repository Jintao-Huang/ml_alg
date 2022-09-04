# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

import time
from typing import Callable, Any, Optional, List, Dict, Union, Tuple, Iterator
import os
from urllib.parse import urljoin
from urllib.error import HTTPError
from urllib.request import urlretrieve
import hashlib
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element
from collections import deque
import logging


__all__ = ["download_files", "calculate_hash", "xml_to_dict", "mywalk"]
#
logger = logging.getLogger(__name__)


def download_files(base_url: str, fnames: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for fname in fnames:
        if '/' in fname:
            dir = os.path.join(save_dir, os.path.dirname(fname))
            os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        if os.path.isfile(save_path):
            continue
        elif os.path.isdir(save_path):
            raise IsADirectoryError(f"save_path: {save_path}")
        # 下载
        file_url = urljoin(base_url, fname)
        logger.info(f"Downloading `{file_url}`")
        try:
            urlretrieve(file_url, save_path)
        except HTTPError:
            raise


if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *

# if __name__ == "__main__":
#     from timeit import timeit
#     def func(x, y):
#         return x @ y
#     x = np.random.randn(1000, 1000)
#     libs_ml.test_time(lambda: func(x, x), 100)
#     libs_ml.test_time(lambda: func(x, x), 100, timer=libs_ml.time_synchronize)
#     print(timeit(lambda: func(x, x), number=100))

# if __name__ == "__main__":
#     import cv2 as cv
#     import numpy as np
#     import torch
#     from torch import Tensor
#     x = np.random.randint(0, 256,(2000, 2000, 3), dtype=np.uint8)
#     def f():
#         return cv.cvtColor(x, cv.COLOR_BGR2RGB)
#     def f2():
#         cv.cvtColor(x, cv.COLOR_BGR2RGB, x)
#         return x
#     libs_ml.test_time(f, 10)
#     libs_ml.test_time(f, 10)


def calculate_hash(fpath: str) -> str:
    """计算文件的hash. 一般用于作为文件名的后缀. e.g. resnet34-b627a593.pth"""
    n = 1024
    sha256 = hashlib.sha256()
    with open(fpath, "rb") as f:
        while True:
            buffer = f.read(n)  # bytes
            if len(buffer) == 0:  # ""
                break
            sha256.update(buffer)
    digest = sha256.hexdigest()
    return digest[:8]

# if __name__ == "__main__":
#     fpath = "/home/jintao/Documents/torch/hub/checkpoints/resnet34-b627a593.pth"
#     print(calculate_hash(fpath))  # b627a593


Node = Dict[str, Union[List["Node"], str]]


def _xml_to_dict(node: Element) -> Node:
    # 树的深搜. 子节点: Dict[str, List]; 根节点: Dict[str, str]
    if len(node) == 0:
        return {node.tag: node.text}
    child = []
    for c in node:
        child.append(_xml_to_dict(c))
    return {node.tag: child}


def xml_to_dict(fpath: str) -> Node:
    # 不处理node中的attribute
    """每个xml node是一个dict, 标题是key, 里面的内容是List. XML_NODE=Dict[str, List[XML_NODE]]"""
    tree = ET.parse(fpath)
    root: Element = tree.getroot()
    return _xml_to_dict(root)


# if __name__ == "__main__":
#     fpath = "asset/1.xml"
#     print(xml_to_dict(fpath))


def _get_folders_fnames(curr_dir: str) -> Tuple[List[str], List[str]]:
    fnames = os.listdir(curr_dir)
    folder_list, fname_list = [], []
    for fname in fnames:
        path = os.path.join(curr_dir, fname)
        if os.path.isdir(path):
            folder_list.append(fname)
        elif os.path.isfile(path):
            fname_list.append(fname)
    return folder_list, fname_list


# level从0开始计数.
Item = Tuple[int, str, List[str], List[str]]  # level, curr_dir, folder_list, fname_list


def mywalk(dir_: str, ignore_dirs: Optional[List[str]] = None) -> Iterator[Item]:
    # 使用广搜. 若遇到ignore_dirs则忽略它及其子文件夹
    # 将每一个文件夹存入队列.
    ignore_dirs: Set[str] = set(ignore_dirs) if ignore_dirs is not None else set()
    dq = deque([dir_])
    level = 0
    while len(dq) > 0:
        dq_len = len(dq)
        for _ in range(dq_len):
            curr_dir = dq.popleft()
            folder_list, fname_list = _get_folders_fnames(curr_dir)
            yield level, curr_dir, folder_list, fname_list
            #
            for folder in folder_list:
                if folder in ignore_dirs:
                    continue
                dq.append(os.path.join(curr_dir, folder))
        level += 1
