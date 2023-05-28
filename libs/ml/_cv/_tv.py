
from ..._types import *
# from libs import *

__all__ = []

def sigmoid_focal_loss(pred: Tensor, target: Tensor, alpha: float=0.25, gamma: float=2.):
    """reduction='none'"""
    prob = torch.sigmoid(pred)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    prob_t = prob * target + (1 - prob) * (1 - target)
    loss.mul_((1 - prob_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss.mul_(alpha_t)
    return loss

if __name__ == "__main__":
    from torchvision.ops.focal_loss import sigmoid_focal_loss as _sigmoid_focal_loss
    x = torch.rand(1000, dtype=torch.float)
    target = torch.randint(0, 2, (1000, ), dtype=torch.float)
    y = ml.test_time(lambda: sigmoid_focal_loss(x, target), 10)
    y2 = ml.test_time(lambda: _sigmoid_focal_loss(x, target), 10)
    print(torch.allclose(y, y2, atol=1e-6))