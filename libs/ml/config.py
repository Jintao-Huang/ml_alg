import os
__all__ = ["RUNS_DIR"]
ROOT_DIR = "/home/jintao/Desktop/coding/python/ml"
RUNS_DIR = os.path.join(ROOT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
