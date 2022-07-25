# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import time
from typing import Callable, Any, Optional, List, Dict, Union
import os
from urllib.parse import urljoin
from urllib.error import HTTPError
from urllib.request import urlretrieve
import numpy as np
import hashlib
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree, Element

__all__ = ["test_time", "download_files", "calculate_hash", "xml_to_dict"]


def test_time(func: Callable[[], Any], number: int = 1, warm_up: int = 0, timer: Optional[Callable[[], float]] = None) -> Any:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    # 预热
    for _ in range(warm_up):
        func()
    #
    res = None
    for _ in range(number):
        t1 = timer()
        res = func()
        t2 = timer()
        ts.append(t2 - t1)
    # 打印平均, 标准差, 最大, 最小
    ts = np.array(ts)
    max_ = ts.max()
    min_ = ts.min()
    mean = ts.mean()
    std = ts.std()
    # print
    print(
        f"time[number={number}]: {mean:.6f}±{std:.6f} |max: {max_:.6f} |min: {min_:.6f}")
    return res


def download_files(base_url: str, fnames: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for fname in fnames:
        if '/' in fname:
            dir = os.path.join(save_dir, os.path.dirname(fname))
            os.makedirs(dir, exist_ok=True)
        save_path = os.path.join(save_dir, fname)
        if os.path.exists(save_path):
            continue
        file_url = urljoin(base_url, fname)
        print(f"Downloading `{file_url}`")
        try:
            urlretrieve(file_url, save_path)
        except HTTPError:
            raise


# if __name__ == "__main__":
#     from timeit import timeit

#     def func(x, y):
#         return x @ y
#     x = np.random.randn(1000, 1000)
#     test_time(lambda: func(x, x), 100)

#     #
#     import sys
#     sys.path.append("/home/jintao/Desktop/coding/python/ml_alg")
#     from libs import libs_ml
#     test_time(lambda: func(x, x), 100, timer=libs_ml.time_synchronize)
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
#     test_time(f, 10)
#     test_time(f, 10)


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
        return {node.tag:node.text}
    child = []
    for c in node:
        child.append(_xml_to_dict(c))
    return {node.tag: child}
    
    

def xml_to_dict(fpath: str) -> Node:
    # 不处理node中的attribute
    """每个xml node是一个dict, 标题是key, 里面的内容是List. XML_NODE=Dict[str, List[XML_NODE]]"""
    tree = ET.parse(fpath)
    root = tree.getroot()  # type: Element
    return _xml_to_dict(root)
    


# if __name__ == "__main__":
#     fpath = "asset/1.xml"
#     print(xml_to_dict(fpath))
    