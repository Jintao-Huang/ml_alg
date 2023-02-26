
from torch.autograd import Function
import torch
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer
from torch.autograd.function import FunctionCtx
from torch import Tensor
from typing import Tuple, List


class Linear(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor, w: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(x, w)
        return x @ w.t() + b

    @staticmethod
    def backward(ctx: FunctionCtx, z_grad: Tensor) ->Tuple[Tensor, Tensor, Tensor]:
        x, w = ctx.saved_tensors
        x_grad = z_grad @ w
        w_grad = z_grad.t() @ x
        b_grad = torch.sum(z_grad, dim=0)
        return x_grad, w_grad, b_grad


class ReLU(Function):
    @staticmethod
    def forward(ctx: FunctionCtx, z: Tensor) ->Tensor:
        z_greater_0 = (z > 0).float()
        ctx.save_for_backward(z_greater_0)
        return z * z_greater_0

    @staticmethod
    def backward(ctx, a_grad: Tensor) -> Tensor:
        z_greater_0, = ctx.saved_tensors
        return a_grad * z_greater_0


class MSELoss(Function):
    @staticmethod
    def forward(ctx, y_pred: Tensor, target: Tensor) -> Tensor:
        ctx.save_for_backward(y_pred, target)
        diff = y_pred - target
        return torch.mean(torch.sum(diff * diff, dim=1))

    @staticmethod
    def backward(ctx, output_grad: Tensor) -> Tuple[Tensor, None]:
        # output_grad: tensor(1., device='cuda:0')
        y_pred, target = ctx.saved_tensors
        N = y_pred.shape[0]
        return output_grad * 2 * (y_pred - target) / N, None


class _SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float) -> None:
        defaults = {"lr": lr}
        super(_SGD, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            params = group['params']
            lr = group['lr']
            for param in params:
                param -= lr * param.grad


def main():
    lr = 1e-1
    # input_c: 1, output_c: 1
    hide_c = 100  # 隐藏层的通道数
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # 1. 制作数据集
    x = torch.linspace(-1, 1, 1000, device=device)[:, None]
    y_true = x *x + 2 + torch.normal(0, 0.1, x.shape, device=device)

    # 2. 参数初始化
    w1 = torch.normal(0, 0.1, (hide_c, 1), requires_grad=True, device=device)
    w2 = torch.normal(0, 0.1, (1, hide_c), requires_grad=True, device=device)
    b1 = torch.zeros((hide_c,), requires_grad=True, device=device)
    b2 = torch.zeros((1,), requires_grad=True, device=device)

    # 3. 训练
    optim = _SGD([w1, w2, b1, b2], lr)
    for i in range(501):
        # 1.forward
        z = Linear().apply(x, w1, b1)
        a = ReLU().apply(z)
        y_pred = Linear().apply(a, w2, b2)
        # 2. loss
        loss = MSELoss().apply(y_pred, y_true)
        # 3. backward
        optim.zero_grad()
        loss.backward()
        # 4.update
        optim.step()
        if i % 10 == 0:
            print(i, "%.6f" % loss)

    # 4. 作图
    x = x.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    plt.scatter(x, y_true, s=20)
    plt.plot(x, y_pred, "r-")
    plt.text(0, 2, "loss %.4f" % loss, fontsize=20, color="r")
    plt.show()


if __name__ == '__main__':
    main()
