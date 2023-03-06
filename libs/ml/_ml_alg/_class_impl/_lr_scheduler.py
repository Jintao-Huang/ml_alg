from ...._types import *


class _LRScheduler:
    def __init__(self, optimizer: Optimizer, last_epoch: int = -1, verbose: bool = False) -> None:

        self.optimizer = optimizer
        for group in optimizer.param_groups:
            if last_epoch == -1:
                group.setdefault("initial_lr", group["lr"])
            else:
                assert "initial_lr" in group

        self.base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        self.last_epoch = last_epoch
        self.verbose = verbose
        self._initial_step()

    def _initial_step(self) -> None:
        self.step()

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != "optimizer"}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def get_lr(self) -> List[float]:
        raise NotImplementedError

    def print_lr(self, is_verbose: bool, group: int, lr: float) -> None:
        if is_verbose:
            print("Adjusting learning rate of group {} to {:.4e}.".format(group, lr))

    def step(self) -> None:
        self.last_epoch += 1
        values = self.get_lr()

        for i, (param_group, lr) in enumerate(zip(self.optimizer.param_groups, values)):
            param_group["lr"] = lr
            self.print_lr(self.verbose, i, lr)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


if __name__ == "__main__":
    # from libs import *
    model = tvm.resnet18().cuda()
    o = optim.SGD(model.parameters(), 0.01, momentum=0.9)
    lr_sc = lrs.CosineAnnealingLR(o, 100, 0.01)
    x = torch.randn((16, 3, 100, 100)).cuda()
    y = model(x).mean()
    y.backward()
    for i in range(100):
        o.step()
        lr_sc.step()
