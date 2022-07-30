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

import re
import requests

__all__ = ["test_time", "download_files",
           "calculate_hash", "xml_to_dict", "update_cite_num"]


def test_time(func: Callable[[], Any], number: int = 1, warm_up: int = 0,
              timer: Optional[Callable[[], float]] = None) -> Any:
    # timer: e.g. time_synchronize
    timer = timer if timer is not None else time.perf_counter
    #
    ts = []
    res = None
    # 预热
    for _ in range(warm_up):
        res = func()
    #
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
        if os.path.isfile(save_path):
            continue
        elif os.path.isdir(save_path):
            raise IsADirectoryError(f"save_path: {save_path}")
        # 下载
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


proxies = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890'
}
headers = {
    "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}


def update_cite_num(fpath: str, ask: bool = True) -> int:
    """这个函数对谷歌学术的人机验证毫无办法. 哈哈哈...
    fname的格式:【{date}】{paper_name}[{c0}].pdf. 
      e.g.【2014_】GloVe[27682].pdf     
          【2103】Transformer in Transformer[295].pdf
    return: 返回0表示成功修改, 返回-1表示失败或未修改
    """
    # 我们需要获取fpath的文章信息, 原始引用信息
    # 随后我们查找第一个, 并获取其引用信息. 如果引用上升, 则更新.
    #  否则打印异常, 并不修改
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"fpath: {fpath}")
    dir_, fname = os.path.split(fpath)  # 返回目录和文件名
    m = re.match(r"【(.+?)】(.+?)\[(\d+?)\]", fname)
    if m is None:
        print(f"异常: fname: {fname}, 请修改合适的文件名")
        return -1
    date, paper_name, n_cite = m.groups()
    params = {
        "q": paper_name,
        "hl": "zh-CN"
    }
    url = "https://scholar.google.com/scholar"
    try:
        req = requests.get(url, proxies=proxies,
                           params=params, headers=headers)
    except Exception as e:
        print(f"获取http异常: {e}")
        return -1
    #
    if req.status_code != 200:
        print(f"req.status_code: {req.status_code}, req.url: {req.url}")
        return -1
    text = req.text
    text = re.sub(r"</?[bi]>", "", text)  # 去掉 <b> <i>等

    #
    pn_list = re.findall(
        r'<a id=".+?" href=".+?" data-clk=".+?" data-clk-atid=".+?">(.+?)</a>', text)
    pn_list = [s for s in pn_list if "[PDF]" not in s]
    c_list = re.findall(r"被引用次数：(\d+)", text)
    assert len(pn_list) == len(c_list)
    pn0 = pn_list[0]
    c0 = c_list[0]
    del pn_list, c_list
    #
    if n_cite > c0:
        print(f"异常: fname: {fname}, pn0: {pn0}, c0: {c0}, 请修改合适的文件名")
        return -1
    #
    new_fname = f"【{date}】{paper_name}[{c0}].pdf"
    print(f'"{fname}" -> "{new_fname}"')
    if ask:
        yn = input(f"    论文名: {pn0}. 是否修改? (y/n)")
    else:
        yn = "y"
    if yn.lower() != "y":
        return -1
    #
    if fname == new_fname:
        print("    引用数未变, 无需修改")
    else:
        new_fpath = os.path.join(dir_, new_fname)
        print("    已修改")
        os.rename(fpath, new_fpath)
    return 0


if __name__ == "__main__":
    update_cite_num(
        "/home/jintao/Desktop/Transformer-xl/Hierarchy/【2103】Transformer in Transformer[295].pdf")
    update_cite_num(
        "/home/jintao/Desktop/2022-4-28[ET]/paper/PTM/【2014_】GloVe[29360].pdf")
