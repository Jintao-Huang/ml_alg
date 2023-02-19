# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from typing import Optional, Tuple, Dict, Callable
from hashlib import sha256
import os
import requests
from torch import Tensor
from torch.nn import Module
import torch
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys
from transformers.configuration_utils import PretrainedConfig
import mini_lightning as ml

__all__ = ["hf_get_state_dict"]
logger = ml.logger


def hf_get_state_dict(hf_home: str, model_id: str, commit_hash: str) -> Dict[str, Tensor]:
    model_id = model_id.replace("/", "--")
    model_fpath = os.path.join(hf_home, "hub", f"models--{model_id}", "snapshots", commit_hash, "pytorch_model.bin")
    state_dict = torch.load(model_fpath)
    return state_dict

