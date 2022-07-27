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


__all__ = ["seed_everything", "time_synchronize", "remove_keys"]


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_deterministic"""
    # 返回seed
    if seed is None:
        seed_min = np.iinfo(np.uint32).min
        seed_max = np.iinfo(np.uint32).max
        seed = random.randint(seed_min, seed_max)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if gpu_dtm is True:
        # https://pytorch.org/docs/stable/backends.html#torch.backends.cudnn.torch.backends.cudnn.benchmark
        # True: cudnn只选择deterministic的卷积算法
        torch.backends.cudnn.deterministic = True
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


# if __name__ == "__main__":
#     # test seed_everything
#     s = seed_everything(3268574154)
#     print(s)
#     print(np.random.random())
#     print()
#     # test time_synchronize
#     x = torch.rand(10000, 10000, device='cuda')
#     y = x @ x
#     t1 = time_synchronize()
#     y = x @ x
#     t2 = time_synchronize()
#     print(t2 - t1)
