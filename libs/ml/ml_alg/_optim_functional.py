# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

__all__ = []

from typing import List, Optional
import torch
from torch import Tensor
import math


@torch.no_grad()
def sgd(
    params: List[Tensor],  # inplace
    d_p_list: List[Tensor],  # copy
    momentum_buffer_list: List[Tensor],  # inplace
    *,
    lr: float,
    momentum: float = 0.,
    dampening: float = 0.,
    weight_decay: float = 0.,
) -> None:
    for i, param in enumerate(params):
        d_p: Tensor = d_p_list[i]
        #
        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)
        #
        if momentum != 0:
            buf: Tensor = momentum_buffer_list[i]
            # buf * momentum + d_p * (1 - dampening)
            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
            #
            d_p = buf
        param.add_(d_p, alpha=-lr)


@torch.no_grad()
def adam(params: List[Tensor],  # inplace
         grads: List[Tensor],  # copy
         exp_avgs: List[Tensor],  # inplace
         exp_avg_sqs: List[Tensor],  # inplace
         state_steps: List[int],  # inplace
         *,
         lr: float = 1e-3,
         beta1: float = 0.9,
         beta2: float = 0.999,
         weight_decay: float = 0.,
         eps: float = 1e-8) -> None:

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        #
        step_t += 1
        state_steps[i] = step_t
        #
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        #
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        #
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t
        exp_avg_hat = exp_avg.div(bias_correction1)
        exp_avg_sq_hat = exp_avg_sq.div(bias_correction2)
        param.addcdiv_(exp_avg_hat, exp_avg_sq_hat.sqrt_().add_(eps), value=-lr)


@torch.no_grad()
def adamw(params: List[Tensor],  # inplace
          grads: List[Tensor],  # copy
          exp_avgs: List[Tensor],  # inplace
          exp_avg_sqs: List[Tensor],  # inplace
          state_steps: List[int],  # inplace
          *,
          lr: float = 1e-3,
          beta1: float = 0.9,
          beta2: float = 0.999,
          weight_decay: float = 1e-2,
          eps: float = 1e-8) -> None:

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        #
        step_t += 1
        state_steps[i] = step_t
        #
        param.mul_(1 - lr * weight_decay)  # 唯一区别
        #
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        #
        bias_correction1 = 1 - beta1 ** step_t
        bias_correction2 = 1 - beta2 ** step_t
        exp_avg_hat = exp_avg.div(bias_correction1)
        exp_avg_sq_hat = exp_avg_sq.div(bias_correction2)
        param.addcdiv_(exp_avg_hat, exp_avg_sq_hat.sqrt_().add_(eps), value=-lr)


if __name__ == "__main__":
    # test sgd
    import torch.nn as nn
    from torch.nn import Module
    from torch.optim.sgd import SGD
    from copy import deepcopy
    momentum = 0.9
    lr = 0.01
    weight_decay = 1e-4
    #
    x = torch.rand(5, 5)
    m: Module = nn.Linear(5, 5)
    m2 = deepcopy(m)
    #
    o = SGD(m.parameters(), lr, momentum, weight_decay=weight_decay)
    for i in range(100):
        loss: Tensor = m(x).mean()
        loss.backward()
        o.step()
    loss1 = loss
    params1 = list(m.parameters())
    ##
    m_buf_list: List[Tensor] = []
    # init
    for p in m2.parameters():
        m_buf_list.append(torch.zeros_like(p))
    #
    for i in range(100):
        loss: Tensor = m2(x).mean()
        loss.backward()
        #
        params = []
        d_p_list = []
        for p in m2.parameters():
            params.append(p)
            d_p_list.append(p.grad)
        sgd(params, d_p_list, m_buf_list, lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss2 = loss
    params2 = list(m2.parameters())
    print(torch.allclose(loss1, loss2, atol=1e-6),
          [torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)])


if __name__ == "__main__":
    # test adam
    import torch.nn as nn
    from torch.nn import Module
    from torch.optim.adam import Adam
    from copy import deepcopy
    lr = 0.01
    weight_decay = 1e-4
    #
    x = torch.rand(5, 5)
    m: Module = nn.Linear(5, 5)
    m2 = deepcopy(m)
    #
    o = Adam(m.parameters(), lr, weight_decay=weight_decay)
    for i in range(100):
        loss: Tensor = m(x).mean()
        loss.backward()
        o.step()
    loss1 = loss
    params1 = list(m.parameters())
    ##
    exp_avgs: List[Tensor] = []
    exp_avg_sqs: List[Tensor] = []
    state_steps: List[int] = []
    # init
    for p in m2.parameters():
        exp_avgs.append(torch.zeros_like(p))
        exp_avg_sqs.append(torch.zeros_like(p))
        state_steps.append(0)
    #
    for i in range(100):
        loss: Tensor = m2(x).mean()
        loss.backward()
        #
        params = []
        grads = []
        for p in m2.parameters():
            params.append(p)
            grads.append(p.grad)
        adam(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr=lr, weight_decay=weight_decay)
    loss2 = loss
    params2 = list(m2.parameters())
    print(torch.allclose(loss1, loss2, atol=1e-6),
          [torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)])

if __name__ == "__main__":
    # test adamw
    import torch.nn as nn
    from torch.nn import Module
    from copy import deepcopy
    from torch.optim.adamw import AdamW
    lr = 0.01
    weight_decay = 1e-4
    #
    x = torch.rand(5, 5)
    m: Module = nn.Linear(5, 5)
    o = AdamW(m.parameters(), lr, weight_decay=weight_decay)
    m2 = deepcopy(m)
    for i in range(100):
        loss: Tensor = m(x).mean()
        loss.backward()
        o.step()
    loss1 = loss
    params1 = list(m.parameters())
    ##
    exp_avgs: List[Tensor] = []
    exp_avg_sqs: List[Tensor] = []
    state_steps: List[int] = []
    # init
    for p in m2.parameters():
        exp_avgs.append(torch.zeros_like(p))
        exp_avg_sqs.append(torch.zeros_like(p))
        state_steps.append(0)
    #
    for i in range(100):
        loss: Tensor = m2(x).mean()
        loss.backward()
        #
        params = []
        grads = []
        for p in m2.parameters():
            params.append(p)
            grads.append(p.grad)
        adamw(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr=lr, weight_decay=weight_decay)
    loss2 = loss
    params2 = list(m2.parameters())
    print(torch.allclose(loss1, loss2, atol=1e-6),
          [torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)])

if __name__ == "__main__":
    # test adam adamw的区别. weight_decay=0时, 没有区别.
    import torch.nn as nn
    from torch.nn import Module
    from copy import deepcopy
    lr = 0.01
    weight_decay = 0
    #
    x = torch.rand(5, 5)
    m: Module = nn.Linear(5, 5)
    m2 = deepcopy(m)
    ##
    exp_avgs: List[Tensor] = []
    exp_avg_sqs: List[Tensor] = []
    state_steps: List[int] = []
    # init
    for p in m.parameters():
        exp_avgs.append(torch.zeros_like(p))
        exp_avg_sqs.append(torch.zeros_like(p))
        state_steps.append(0)
    #
    for i in range(100):
        loss: Tensor = m(x).mean()
        loss.backward()
        #
        params = []
        grads = []
        for p in m.parameters():
            params.append(p)
            grads.append(p.grad)
        adam(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr=lr, weight_decay=weight_decay)
    loss1 = loss
    params1 = list(m.parameters())
    ##
    exp_avgs: List[Tensor] = []
    exp_avg_sqs: List[Tensor] = []
    state_steps: List[int] = []
    # init
    for p in m2.parameters():
        exp_avgs.append(torch.zeros_like(p))
        exp_avg_sqs.append(torch.zeros_like(p))
        state_steps.append(0)
    #
    for i in range(100):
        loss: Tensor = m2(x).mean()
        loss.backward()
        #
        params = []
        grads = []
        for p in m2.parameters():
            params.append(p)
            grads.append(p.grad)
        adamw(params, grads, exp_avgs, exp_avg_sqs, state_steps, lr=lr, weight_decay=weight_decay)
    loss2 = loss
    params2 = list(m2.parameters())
    print(torch.allclose(loss1, loss2, atol=1e-6),
          [torch.allclose(p1, p2, atol=1e-6) for p1, p2 in zip(params1, params2)])
