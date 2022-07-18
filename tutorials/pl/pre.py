import sys
sys.path.append("/home/jintao/Desktop/coding/python/ml_alg")
from libs import *
# 
_ROOT_DIR = "/home/jintao/Desktop/coding/python/ml_alg"
RUNS_DIR = os.path.join(_ROOT_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
PL_RUNS_DIR = os.path.join(RUNS_DIR, "PL")