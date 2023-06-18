# Author: Jintao Huang
# Email: huangjintao@mail.ustc.edu.cn
# Date:

from ..._types import *

class _CosineAnnealingLR(LRScheduler):
    """
    epoch=0: lr=initial_lr
    epoch=T_max: lr=eta_min
    """

    def __init__(self, optimizer: Optimizer, T_max: int, eta_min: float = 0., last_epoch: int = -1) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self, last_epoch=None) -> List[float]:
        if last_epoch is None:
            last_epoch = self.last_epoch
        return ml.cosine_annealing_lr(last_epoch, self.T_max, self.eta_min, self.base_lrs)


#
"""
Note! In order to avoid LR to be 0 in the first step, we shifted one step to the left
iter_idx=-1: lr=0 or cosine_annealing_lr(0) * 0
iter_idx=warmup-1: lr=cosine_annealing_lr(warmup)
iter_idx=T_max-1: lr=eta_min or cosine_annealing_lr(T_max)
"""
# WarmupCosineAnnealingLR


class WarmupCosineAnnealingLR2(_CosineAnnealingLR):
    """Note! In order to avoid LR to be 0 in the first step, we shifted one step to the left
    iter_idx=-1: lr=0
    iter_idx=warmup-1: lr=initial_lr or cosine_annealing_lr(0)
    iter_idx=warmup+T_max-1: lr=eta_min or cosine_annealing_lr(T_max)
    """

    def __init__(self, optimizer: Optimizer, warmup: int, T_max: int, eta_min: float = 0.,
                 last_epoch: int = -1) -> None:
        self.warmup = warmup
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self) -> List[float]:
        last_epoch = self.last_epoch + 1  # shifted one step to the left
        if last_epoch < self.warmup:
            # warmup
            scale = last_epoch / self.warmup
            return [lr * scale for lr in self.base_lrs]
        lrs = super().get_lr(last_epoch - self.warmup)
        return lrs


if __name__ == "__main__":
    import unittest as ut
    import torch
    from torch.optim.sgd import SGD
    from torch.nn.parameter import Parameter
    from torch.optim.lr_scheduler import CosineAnnealingLR

    def allclose(lr: float, lr2: float, atol=1e-6) -> Tuple[bool, float]:
        """return bool, true_atol: float"""
        atol2 = abs(lr - lr2)
        res = False
        if atol2 < atol:
            res = True
        return res, atol2

    class TestLrs(ut.TestCase):
        def test_warmup2(self):
            initial_lr = 1e-2
            T_max = 7
            eta_min = 1e-4
            max_epoch = 20
            #
            optim = SGD([Parameter(torch.randn(100,))], initial_lr)
            warmup = 3
            lr_s = WarmupCosineAnnealingLR2(optim, warmup, T_max, eta_min)
            for i in range(max_epoch):
                lr = lr_s.get_last_lr()[0]
                if i == 0:
                    self.assertTrue(lr > 0)
                elif i == warmup - 2:
                    self.assertTrue(lr != initial_lr)
                elif i >= warmup - 1:
                    lr2 = ml.cosine_annealing_lr(i + 1 - warmup, T_max, eta_min, [initial_lr])[0]
                    b, atol = allclose(lr, lr2)
                    self.assertTrue(b, msg=f"atol: {atol}")
                    if i == warmup - 1:
                        self.assertTrue(lr == initial_lr)
                    elif i == T_max + warmup - 1:
                        self.assertTrue(lr == eta_min)
                optim.step()
                lrs.step()
    ut.main()
