from ._loss import *
from ._layers import *
from ._activations import *
from ._vision import *
from ._sparse import *
from ._utils import *
# Ref: https://pytorch.org/docs/stable/nn.functional.html
__all__ = []


"""
note: 这里写的是forward. 在backward时可能会报错(因为inplace关系). 
    若要用于backward, 可以取消所有inplace操作
    或参考https://github.com/pytorch/pytorch/blob/master/tools/autograd/derivatives.yaml
    (有些函数的backward需要用到result, 则不能对result进行inplace操作等). e.g. sqrt_, rsqrt_, sigmoid_, exp_
    错误样例见下
"""

# if __name__ == "__main__":
#     import torch
#     # 1. x.mul(a)中的梯度为 2a. 所以a不能inplace.
#     x_ = torch.tensor([1.], requires_grad=True)
#     x = x_.clone()
#     a = torch.tensor(2)
#     x = x.mul(a)
#     a.mul_(100)
#     x.backward()
#     print(x_.grad)
#     # 2. x.sqrt()的梯度为 grad / (2 * result.conj()). 所以result不能修改
#     x_ = torch.tensor([1.], requires_grad=True)
#     x = x_.clone()
#     result = x.sqrt()
#     result.mul_(100)
#     result.backward()
#     print(x_.grad)


# 用于学习, 速度慢于F.
# 没有显示说明, 就不是inplace的