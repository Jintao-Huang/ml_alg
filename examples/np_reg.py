import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from typing import Tuple, List


def fc_forward(x: ndarray, w: ndarray, b: ndarray) -> ndarray:
    """
    x: [N, In]
    w: [Out, In]
    b: [Out]
    return: [N, Out]
    """
    return x @ w.T + b


def fc_backward(x: ndarray, w: ndarray, z_grad: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    """
    x: [N, In]
    w: [Out, In]
    z_grad: [N, Out]
    return: Tuple[x_grad, w_grad, b_grad]
        x_grad: [N, In]
        w_grad: [Out, In]
        b_grad: [Out]
    """
    x_grad = z_grad @ w
    w_grad = z_grad.T @ x
    b_grad = np.sum(z_grad, axis=0)

    return x_grad, w_grad, b_grad


def relu_forward(z: ndarray) -> ndarray:
    """relu前向
    z: [Out]
    return: [Out]
    """
    return z * (z > 0)


def relu_backward(z: ndarray, a_grad: ndarray) -> ndarray:
    """relu反向
    z: [Out]
    a_grad: [Out]
    return: [Out]
    """
    return a_grad * (z > 0)


def mse_loss_forward(y_true: ndarray, y_pred: ndarray) -> float:
    """
    y_true: [N, Out]
    y_pred: [N, Out]
    return: []
    """
    diff = y_true - y_pred
    return np.mean(np.sum(diff * diff, axis=1))


def mse_loss_backward(y_true: ndarray, y_pred: ndarray) -> ndarray:
    """
    y_true: [N, Out]
    y_pred: [N, Out]
    return: [N, Out]
    """
    N = y_true.shape[0]
    return 2 * (y_pred - y_true) / N


def sgd(params: List[ndarray], grads: List[ndarray], lr: float = 1e-2) -> None:
    for i in range(len(params)):
        params[i] -= lr * grads[i]


if __name__ == '__main__':
    lr = 1e-1
    # input_c: 1, output_c: 1
    hide_c = 100  # 隐藏层的通道数

    # 1. 制作数据集
    rng = np.random.default_rng(42)
    x = np.linspace(-1, 1, 1000)[:, None]
    y_true = x * x + 2 + rng.normal(0, 0.1, x.shape)

    # 2. 参数初始化.
    w1 = rng.normal(0, 0.1, (hide_c, x.shape[1]))
    w2 = rng.normal(0, 0.1, (y_true.shape[1], hide_c))
    b1 = np.zeros((hide_c,))
    b2 = np.zeros((1,))

    # 3. 训练
    for i in range(501):
        # 1.forward
        z = fc_forward(x, w1, b1)
        a = relu_forward(z)
        y_pred = fc_forward(a, w2, b2)

        # 2. loss
        loss = mse_loss_forward(y_true, y_pred)

        # 3. backward
        pred_grad = mse_loss_backward(y_true, y_pred)
        a_grad, w2_grad, b2_grad = fc_backward(a, w2, pred_grad)
        z_grad = relu_backward(z, a_grad)
        _, w1_grad, b1_grad = fc_backward(x, w1, z_grad)

        # 4.update
        params = [w1, w2, b1, b2]
        grads = [w1_grad, w2_grad, b1_grad, b2_grad]
        sgd(params, grads, lr)
        if i % 10 == 0:
            print(i, "%.6f" % loss)

    # 4. 作图
    plt.scatter(x, y_true, s=20)
    plt.plot(x, y_pred, "r-")
    plt.text(0, 2, "loss %.4f" % loss, fontsize=20, color="r")
    plt.show()
