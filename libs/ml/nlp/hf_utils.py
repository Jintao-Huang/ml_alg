from typing import Optional, Tuple, Dict, Callable
from hashlib import sha256
import os
import requests
from torch import Tensor
from torch.nn import Module
import torch
from torch.nn.modules.module import _IncompatibleKeys as IncompatibleKeys

__all__ = ["hf_hash", "get_hf_fname", "hf_load_state_dict", "replace_callback"]

HEADERS = {
    # 'user-agent':
    # 'transformers/4.19.2; python/3.9.12; session_id/f3ef440752714dc7931cd95cd605a60e; torch/1.12.1+cu116; file_type/config; from_auto_class/True'
}

# 代理
PROXIES = {
    'http': '127.0.0.1:7890',
    'https': '127.0.0.1:7890'
}

HF_HOME = os.environ.get("HF_HOME", None)
HF_URL = 'https://huggingface.co/{model_id:s}/resolve/{revision:s}/{filename:s}'
assert(HF_HOME is not None)


def hf_hash(url: str, etag: Optional[str] = None) -> str:
    url_b: bytes = url.encode("utf-8")
    fname: str = sha256(url_b).hexdigest()
    if etag is not None:
        etag_b: bytes = etag.encode("utf-8")
        fname += "." + sha256(etag_b).hexdigest()
    return fname


def get_hf_fname(model_id: str) -> Tuple[str, str]:
    """返回config, model的fname"""

    cache_dir = os.path.join(HF_HOME, "transformers")
    #
    revision = "main"
    config_url = HF_URL.format(model_id=model_id, revision=revision, filename="config.json")
    model_url = HF_URL.format(model_id=model_id, revision=revision, filename="pytorch_model.bin")
    #
    r = requests.head(config_url, headers=HEADERS, allow_redirects=False, proxies=PROXIES, timeout=10)
    etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
    config_fname = hf_hash(config_url, etag)
    config_path = os.path.join(cache_dir, config_fname)
    #
    r = requests.head(model_url, headers=HEADERS, allow_redirects=False, proxies=PROXIES, timeout=10)
    etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
    model_fname = hf_hash(model_url, etag)
    model_path = os.path.join(cache_dir, model_fname)
    return config_path, model_path


def hf_load_state_dict(model: Module, state_dict: Dict[str, Tensor], prefix_key: str = "",
                       callback_func: Callable[[Dict[str, Tensor]], Dict[str, Tensor]] = None) -> IncompatibleKeys:
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
