# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ..._types import *

__all__ = ["hf_get_state_dict"]
logger = ml.logger


def hf_get_state_dict(hf_home: str, model_id: str, commit_hash: str) -> Dict[str, Tensor]:
    model_id = model_id.replace("/", "--")
    model_fpath = os.path.join(hf_home, "hub", f"models--{model_id}", "snapshots", commit_hash, "pytorch_model.bin")
    state_dict = torch.load(model_fpath)
    return state_dict

