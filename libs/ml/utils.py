# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import random
import torch
import numpy as np
import torch.cuda as cuda
import time
from typing import Optional, Callable, Tuple, List, Dict, Any
from torch import Tensor
from collections import defaultdict


__all__ = ["seed_everything", "time_synchronize",
           "remove_keys", "gen_seed_list", "multi_runs"]


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_deterministic"""
    # 返回seed
    if seed is None:
        # seed_min = np.iinfo(np.uint32).min
        seed_max = np.iinfo(np.uint32).max
        seed = random.randint(0, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if gpu_dtm is True:
        # https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        # True: cudnn只选择deterministic的卷积算法
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True)  # 会报错
        # True: cuDNN从多个卷积算法中进行benchmark, 选择最快的
        # 若deterministic=True, 则benchmark一定为False
        torch.backends.cudnn.benchmark = False
    print(f"Global seed set to {seed}")
    return seed


def time_synchronize() -> float:
    # 单位: 秒
    cuda.synchronize()
    return time.perf_counter()


def remove_keys(state_dict: Dict[str, Any], prefix_keys: List[str]) -> Dict[str, Any]:
    """不是inplace的"""
    res = {}
    for k, v in state_dict.items():
        need_saved = True
        for pk in prefix_keys:
            if k.startswith(pk):
                need_saved = False
                break
        if need_saved:
            res[k] = v
    return res

if __name__ == "__main__":
    import sys
    import os
    _ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
    if not os.path.isdir(_ROOT_DIR):
        raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
    sys.path.append(_ROOT_DIR)
    from libs import *

# if __name__ == "__main__":
#     # test seed_everything
#     s = seed_everything(3234335211)
#     print(s)
#     # test time_synchronize
#     x = torch.randn(10000, 10000, device='cuda')
#     res = libs_utils.test_time(lambda: x@x, 10, 0, time_synchronize)
#     print(res[1, :100])

def gen_seed_list(n: int, seed: Optional[int] = None,) -> List[int]:
    max_ = np.iinfo(np.uint32).max
    random_state = np.random.RandomState(seed)
    return random_state.randint(0, max_, n).tolist()


def multi_runs(collect_res: Callable[[int], Dict[str, float]], n: int, seed: Optional[int] = None, *,
               seed_list: Optional[List[int]] = None) -> Dict[str, Dict[str, Any]]:
    """跑n次的结果.
    collect_res: 函数: 传入seed, 返回result.
    n: 跑的次数. {seed_list的优先级更高, 若提供seed_list, 则n, seed无效}
    """
    t = time.perf_counter()
    if seed_list is None:
        seed_list = gen_seed_list(n, seed)
    n = len(seed_list)
    result: Dict[str, List] = defaultdict(list)
    for _seed in seed_list:
        _res = collect_res(_seed)
        for k, v in _res.items():
            result[k].append(v)
    t = int(time.perf_counter() - t)
    h, m, s = t // 3600, t // 60 % 60, t % 60
    t = f"{h:02d}:{m:02d}:{s:02d}"
    # 计算mean, std等.
    res: Dict[str, Dict[str, Any]] = {}
    res_str: List = []
    for k, v_list in result.items():
        v_list = np.array(v_list)
        mean = v_list.mean()
        std = v_list.std()
        max_ = v_list.max()
        min_ = v_list.min()
        res_str.append(
            f"{k}[n={n}]: {mean:.6f}±{std:.6f} |max: {max_:.6f} |min: {min_:.6f}| time: {t}| seed_list: {seed_list}")
        res[k] = {
            "n": n,
            "mean": mean,
            "std": std,
            "max_": max_,
            "min_": min_,
            "time": t,
            "seed_list": seed_list,
        }
    print("\n".join(res_str))
    return res
