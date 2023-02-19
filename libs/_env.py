import os
# 以下环境变量需要用户自定义设置, 这里为了自己方便进行导入
TORCH_HOME = os.environ.get("TORCH_HOME", None)
DATASETS_PATH = os.environ.get("DATASETS_PATH", "./.dataset")
HF_HOME = os.environ.get("HF_HOME", None)
CACHE_HOME = os.environ.get("CACHE_HOME", "./.cache")
