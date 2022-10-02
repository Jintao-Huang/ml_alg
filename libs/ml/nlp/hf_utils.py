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

__all__ = ["model_from_pretrained", "hf_load_state_dict", "replace_callback"]
logger = ml.logger


def model_from_pretrained(model_type: type, hf_home: str, model_id: str, config: PretrainedConfig, replace_keys: Dict[str, str]) -> Module:
    commit_hash = config._commit_hash
    model_fpath = os.path.join(hf_home, "hub", f"models--{model_id}", "snapshots", commit_hash, "pytorch_model.bin")
    state_dict = torch.load(model_fpath)
    model: Module = model_type(config)
    logger.info(hf_load_state_dict(model, state_dict, "", replace_callback(replace_keys)))
    return model


def hf_load_state_dict(model: Module, state_dict: Dict[str, Tensor], prefix_key: str = "",
                       callback_func: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None) -> IncompatibleKeys:
    """先fix, 再replace, 再prefix"""
    def _fix_keys(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.endswith("LayerNorm.gamma"):
                k = k.replace("gamma", "weight")
            elif k.endswith("LayerNorm.beta"):
                k = k.replace("beta", "bias")
            new_state_dict[k] = v
        return new_state_dict
    state_dict = _fix_keys(state_dict)
    # 额外的操作.
    if callback_func:
        state_dict = callback_func(state_dict)
    # prefix
    if prefix_key != "":
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[prefix_key + k] = v
        state_dict = new_state_dict
    #
    return model.load_state_dict(state_dict, strict=False)


def replace_callback(replace_keys: Dict[str, str]) -> Callable[[Dict[str, Tensor]], Dict[str, Tensor]]:
    # e.g. 将state_dict的keys进行替换.
    def func(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for k, v in replace_keys.items():
            state_dict[v] = state_dict[k]
            state_dict.pop(k)
        return state_dict
    return func
