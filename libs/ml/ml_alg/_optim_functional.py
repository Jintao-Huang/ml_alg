# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

__all__ = []

from typing import List, Optional
import torch
from torch.optim.sgd import SGD, sgd as _sgd
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from copy import deepcopy


@torch.no_grad()
def sgd(
    params: List[Tensor],  # inplace
    d_p_list: List[Tensor],  # copy
    momentum_buffer_list: List[Optional[Tensor]],  # inplace
    *,
    lr: float,
    momentum: float = 0.,
    dampening: float = 0.,
    weight_decay: float = 0.,
    nesterov: bool = False,
) -> None:
    for i, param in enumerate(params):
        d_p: Tensor = d_p_list[i].clone()
        #
        if weight_decay != 0:
            d_p = d_p.add_(param, alpha=weight_decay)

        if momentum != 0:
            buf: Optional[Tensor] = momentum_buffer_list[i]
            if buf is None:
                buf = torch.clone(d_p)
                momentum_buffer_list[i] = buf
            else:
                # buf * momentum + d_p * (1 - dampening)
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            # 
            if nesterov:  # [?]
                d_p = d_p.add_(buf, alpha=momentum)  # buf * momentum + d_p
            else:
                d_p = buf

        param.sub_(d_p, alpha=lr)


if __name__ == "__main__":
    momentum = 0.9
    lr = 0.01
    weight_decay = 1e-4
    nesterov = True
    #
    x = torch.rand(5, 5)
    m: Module = nn.Linear(5, 5)
    o = SGD(m.parameters(), lr, momentum, weight_decay=weight_decay)
    m2 = deepcopy(m)
    for i in range(10):
        loss: Tensor = m(x).mean()
        loss.backward()
        o.step()
    print(list(m.parameters()), loss)
    #
    n_param = len(list(m.parameters()))
    m_buf_list: List[Optional[Tensor]] = [None] * n_param
    for i in range(10):
        loss: Tensor = m2(x).mean()
        loss.backward()
        params = []
        d_p_list = []
        for p in m2.parameters():
            params.append(p)
            d_p_list.append(p.grad)
        sgd(params, d_p_list, m_buf_list, lr=lr, momentum=momentum, weight_decay=weight_decay)
    print(list(m.parameters()), loss)