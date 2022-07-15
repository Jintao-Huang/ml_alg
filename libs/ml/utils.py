import random
import torch
import numpy as np
import torch.cuda as cuda
import time
from typing import Optional


def seed_everything(seed: Optional[int] = None, gpu_dtm: bool = False) -> int:
    """gpu_dtm: gpu_determinstic"""
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
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    return seed


def time_synchronize() -> float:
    # 单位: 秒
    cuda.synchronize()
    return time.perf_counter()


if __name__ == "__main__":
    # test seed_everything
    s = seed_everything(3268574154)
    print(s)
    print(np.random.random())
    print()
    # test time_synchronize
    x = torch.rand(10000, 10000, device='cuda')
    y = x @ x
    t1 = time_synchronize()
    y = x @ x
    t2 = time_synchronize()
    print(t2 - t1)
    #
