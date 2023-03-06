from ..._types import *
from .._ml_alg._metrics import pairwise_cosine_similarity


__all__ = ["GatherLayer", "NT_Xent_loss"]


class GatherLayer(Function):
    """ref: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py"""

    @staticmethod
    def forward(ctx: FunctionCtx, x: Tensor) -> Tuple[Tensor]:
        res = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(res, x)
        return tuple(res)

    @staticmethod
    def backward(ctx: FunctionCtx, *grads: Tensor) -> Tensor:
        res = grads[dist.get_rank()]
        res *= dist.get_world_size()  # for same grad with 2 * batch_size; mean operation in ddp across device.
        return res


def NT_Xent_loss(features: Tensor, temperature: float = 0.1) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    features: Tensor[float]. [2N, E]
    return: loss: [], pos_idx_mean: [2N], acc: [2N], acc_5: [2N]
    """
    NINF = -torch.inf
    device = features.device
    N = features.shape[0] // 2
    cos_sim = pairwise_cosine_similarity(features, features)
    cos_sim = cos_sim / temperature
    self_mask = torch.arange(2 * N, dtype=torch.long, device=device)
    pos_mask = self_mask.roll(N)  # [2N]
    cos_sim[self_mask, self_mask] = NINF
    pos_sim = cos_sim[self_mask, pos_mask]
    #
    loss = -pos_sim + torch.logsumexp(cos_sim, dim=-1)
    loss = loss.mean()
    #
    pos_sim = pos_sim.clone().detach_()[:, None]  # [2N, 1]
    cos_sim = cos_sim.clone().detach_()  # [2N, 2N]
    cos_sim[self_mask, pos_mask] = NINF
    comb_sim = torch.concat([pos_sim, cos_sim], dim=-1)
    # pos_sim在哪一位, 即idx/order.
    pos_idx = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)  # 最后两位是NINF(即忽略)
    acc = (pos_idx == 0).float()
    acc_5 = (pos_idx < 5).float()
    return loss, pos_idx, acc, acc_5
