import sys
import os
_ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
if not os.path.isdir(_ROOT_DIR):
    raise IOError(f"_ROOT_DIR: {_ROOT_DIR}")
sys.path.append(_ROOT_DIR)
from libs import *

RUNS_DIR = os.path.join(_ROOT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
