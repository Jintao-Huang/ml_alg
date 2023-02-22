"""
cls_token_id=0, pad_token_id=1
"""

from torch.nn import Module
from typing import Tuple, Dict, Union, List, Optional
import math
import torch
from torch import Tensor, dtype as Dtype, device as Device
import torch.nn as nn
import torch.nn.functional as F


def create_pos_ids(input_ids: Tensor, padding_idx: int = 1) -> Tensor:
    """input_ids: [N, L]"""
    mask = input_ids != padding_idx  # choice: m=1
    pos_ids = torch.cumsum(mask, dim=1).mul_(mask).to(dtype=torch.long)
    pos_ids.add_(padding_idx)
    return pos_ids


if __name__ == "__main__":
    input_ids = torch.tensor([
        [0,  9064,  6406,   786,    12, 48921, 19830,    19,    49,   737,
         1530,     4, 21355,  1530,  1642,    23, 39723,  1215,   808,  1178,
         2744,   134,     4,   221, 30547, 19830,    32,  8266,     4,   152,
         16, 10639,    31,  2105, 47762,    18, 22209, 49320,     4, 19746,
         1215, 11474,  8237, 49024,     2],
        [0,  9064,  6406,   786,    12, 48921, 19830,    19,    49,   737,
         1530,     4,  1437,     2,     1,     1,     1,     1,     1,     1,
         1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
         1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
         1,     1,     1,     1,     1]
    ])


# if __name__ == "__main__":
#     print(create_pos_ids(input_ids))


class RobertaEmbeddings(Module):
    def __init__(
            self,
            vocab_size: int = 50265,
            hidden_size: int = 768,
            max_pos_embeddings: int = 514,
            pad_token_id: int = 1,
            layer_norm_eps: float = 1e-5,
            dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.padding_idx = pad_token_id
        # self.pos_embedding_type = "absolute"
        #
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.pos_embeddings = nn.Embedding(max_pos_embeddings, hidden_size, padding_idx=pad_token_id)
        #
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, input_ids: Tensor
    ) -> Tensor:
        """input_ids: [N, L]"""
        pos_ids = create_pos_ids(input_ids, self.padding_idx)
        res: Tensor = self.word_embeddings(input_ids)
        res.add_(self.pos_embeddings(pos_ids))
        #
        res = self.LayerNorm(res)
        res = self.dropout(res)
        return res


# if __name__ == "__main__":
#     embedding = RobertaEmbeddings()
#     res = embedding(input_ids)
#     print(res)


class RobertaSelfAttention(Module):
    def __init__(
        self,
        hidden_size: int = 768,
        n_attn_heads: int = 12,
        attn_dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        assert hidden_size % n_attn_heads == 0
        self.n_attn_heads = n_attn_heads
        self.hidden_size = hidden_size
        #
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        #
        self.dropout = nn.Dropout(attn_dropout_p)

    def _transpose_to_head(self, x: Tensor) -> Tensor:
        """
        x: [N, L, E]
        return: [N, H, L, E//H]
        """
        N, L, E = x.shape
        H = self.n_attn_heads
        new_x_shape = N, L, H, E//H
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def _transpose_from_head(self, x: Tensor) -> Tensor:
        """
        x: [N, H, L, E//H]
        return: [N, L, E]
        """
        N, _, L, _ = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = N, L, self.hidden_size
        return x.view(new_x_shape)

    def forward(
        self,
        x: Tensor,  # [N, L, E]
        attn_mask: Tensor,  # [N, L], 0 / -inf
        output_attn: bool = False,
    ) -> Tuple[Tensor, ...]:
        """
        return: output: [N, L, E], attn_dist: [N, H, L, L]
        """
        K = self._transpose_to_head(self.key(x))  # [N, L, E] -> [N, H, L, E//H]
        V = self._transpose_to_head(self.value(x))
        Q = self._transpose_to_head(self.query(x))
        E, H = self.hidden_size, self.n_attn_heads
        #
        Q.div_(math.sqrt(E//H))
        attn_scores = Q @ K.transpose(-1, -2)  # [N, H, L, L]
        # attn_scores.div_(math.sqrt(E//H))  # 已经对Q进行了操作.
        attn_mask = attn_mask[:, None, None, :]
        attn_scores.add_(attn_mask)  # [N, 1, 1, L]. for pad
        #
        attn_dist = F.softmax(attn_scores, dim=-1)  # 分布
        attn_dist = self.dropout(attn_dist)  # [N, H, L, L]
        output: Tensor = attn_dist @ V  # [N, H, L, E//H]
        #
        output = self._transpose_from_head(output)  # [N, L, H, E//H] -> [N, L, E]
        return (output, ) if not output_attn else (output, attn_dist)


class RobertaSelfOutput(Module):
    def __init__(self, hidden_size: int = 768, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        """接在self attention后面"""
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + x0)
        return x


class RobertaAttention(Module):
    def __init__(self, hidden_size: int = 768, n_attn_heads: int = 12,
                 attn_dropout_p: float = 0.1, layer_norm_eps: float = 1e-5,
                 dropout_p: float = 0.1) -> None:
        super().__init__()
        self.attn = RobertaSelfAttention(hidden_size, n_attn_heads, attn_dropout_p)
        self.output = RobertaSelfOutput(hidden_size, layer_norm_eps, dropout_p)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        output_attn: bool = False,
    ) -> Tuple[Tensor, ...]:
        x0 = x
        res = self.attn(x, attn_mask, output_attn)
        x = self.output(res[0], x0)
        return (x, *res[1:])


class RobertaIntermediate(Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = F.gelu(x)
        return x


class RobertaOutput(Module):
    """接在RobertaIntermediate后"""

    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        x = self.LayerNorm(x + x0)
        return x


class RobertaLayer(Module):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072, n_attn_heads: int = 12,
                 attn_dropout_p: float = 0.1, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.attn = RobertaAttention(hidden_size, n_attn_heads, attn_dropout_p, layer_norm_eps, dropout_p)
        self.intermediate = RobertaIntermediate(hidden_size, intermediate_size)
        self.output = RobertaOutput(hidden_size, intermediate_size, layer_norm_eps, dropout_p)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        output_attn: bool = False,
    ) -> Tuple[Tensor, ...]:
        res = self.attn(x, attn_mask, output_attn)
        x0 = x = res[0]
        x = self.intermediate(x)
        x = self.output(x, x0)
        #
        return (x, *res[1:])


class RobertaEncoder(Module):
    def __init__(self, n_layers: int = 12,
                 hidden_size: int = 768, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 gradient_checkpoint: bool = False) -> None:
        super().__init__()
        self.gradient_checkpoint = gradient_checkpoint
        self.layer = nn.ModuleList([RobertaLayer(hidden_size, intermediate_size, n_attn_heads, attn_dropout_p,
                                                 layer_norm_eps, dropout_p) for _ in range(n_layers)])

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        attn_list: List[Tensor] = []
        for _, layer_module in enumerate(self.layer):
            if self.gradient_checkpoint:
                x_tuple = checkpoint(layer_module, x, attn_mask, output_attn)
            else:
                x_tuple = layer_module(x, attn_mask, output_attn)
            #
            if output_attn:
                attn_list.append(x_tuple[1])
            x = x_tuple[0]
        res = {"output": x}
        if output_attn:
            res["attn_dist"] = torch.stack(attn_list, dim=0)
        return res


class RobertaPool(Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, L, E]"""
        x = x[:, 0]
        x = self.dense(x)
        x = torch.tanh(x)
        return x


class RobertaLMHead(Module):
    def __init__(self, hidden_size: int = 768, layer_norm_eps: float = 1e-5, vocab_size: int = 50265) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dense2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [N, L, E]
        return: [N, L, V]
        """
        x = self.dense(x)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        return x


class RobertaPreTrainedModel(Module):
    def __init__(self, init_range: Optional[float] = 0.02, module_name: str = "") -> None:
        super().__init__()
        self.module_name = module_name
        self.init_range = init_range

    def post_init(self) -> None:
        if self.init_range is not None:
            for m in self.modules():
                self._init_weights(m)
        #
        oe = self._get_output_embedding()
        if oe is not None:
            io = self._get_input_embedding()
            oe.weight = io.weight

    def _get_output_embedding(self) -> Optional[Module]:
        return

    def _get_input_embedding(self) -> Module:
        module = self
        if self.module_name != "":
            module = getattr(self, self.module_name)
        return module.embeddings.word_embeddings

    def _init_weights(self, m: Module) -> None:
        assert self.init_range is not None
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                m.weight.normal_(mean=0.0, std=self.init_range)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.normal_(mean=0.0, std=self.init_range)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.zero_()
                m.weight.fill_(1.0)


class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, vocab_size: int = 50265, hidden_size: int = 768, max_pos_embeddings: int = 514,
                 pad_token_id: int = 1, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 n_layers: int = 12, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1,
                 add_pooling_layer: bool = True, init_range: Optional[float] = 0.02,
                 gradient_checkpoint: bool = False) -> None:
        super().__init__(init_range)
        self.embeddings = RobertaEmbeddings(vocab_size, hidden_size, max_pos_embeddings,
                                            pad_token_id, layer_norm_eps, dropout_p)
        self.encoder = RobertaEncoder(n_layers, hidden_size, intermediate_size,
                                      n_attn_heads, attn_dropout_p, layer_norm_eps, dropout_p, gradient_checkpoint)
        self.pool = None
        if add_pooling_layer:
            self.pool = RobertaPool(hidden_size)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        """
        input_ids: [N, L]
        attn_mask: [N, L]
        return: output: [N, L, E], pool_output: [N, E], attn_dist: [NL, N, H, L, L]
        """
        res = {}
        dtype = self.embeddings.word_embeddings.weight.dtype
        attn_mask = self._create_attn_mask(attn_mask, dtype)
        x = self.embeddings(input_ids)
        res = self.encoder(x, attn_mask, output_attn)
        x = res["output"]
        if self.pool is not None:
            pool_output = self.pool(x)
            res["pool_output"] = pool_output
        #
        return res

    @staticmethod
    def _create_attn_mask(attn_mask: Tensor, dtype: Dtype = torch.float32) -> Tensor:
        """
        attn_mask: [N, L], long
        return: [N, 1, 1, L]. float
        """
        attn_mask = attn_mask.to(dtype)
        NINF = torch.finfo(dtype).min
        attn_mask = (1.0 - attn_mask) * NINF
        return attn_mask


class RobertaForMLM(RobertaPreTrainedModel):
    def __init__(self, vocab_size: int = 50265, hidden_size: int = 768, max_pos_embeddings: int = 514,
                 pad_token_id: int = 1, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 n_layers: int = 12, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1,
                 add_pooling_layer: bool = True,
                 init_range: Optional[float] = 0.02, gradient_checkpoint: bool = False) -> None:
        super().__init__(init_range, "roberta")
        self.vocab_size = vocab_size
        self.roberta = RobertaModel(vocab_size, hidden_size, max_pos_embeddings, pad_token_id, layer_norm_eps, dropout_p,
                                    n_layers, intermediate_size, n_attn_heads, attn_dropout_p, add_pooling_layer, None, gradient_checkpoint)
        self.lm_head = RobertaLMHead(hidden_size, layer_norm_eps, vocab_size)
        self.post_init()

    def _get_output_embedding(self) -> Optional[Module]:
        return self.lm_head.dense2

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        labels: Optional[Tensor] = None,
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        """
        return: {output, pool_output; mlm_loss, mlm_acc, attn_dist}
        """
        res = self.roberta(input_ids, attn_mask, output_attn)
        z = res["output"]
        logits: Tensor = self.lm_head(z)  # 未过softmax

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # ignore_index = -100
            res["mlm_loss"] = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            #
            logits = logits.detach()
            masked_id = torch.nonzero(labels != -100, as_tuple=True)
            y_pred = logits[masked_id].argmax(-1)
            y_label = labels[masked_id]
            res["mlm_acc"] = libs_ml.accuracy(y_pred, y_label, "multiclass", -1)
        return res


if __name__ == "__main__":
    from libs import *
    libs_ml.select_device([0])
    model_id = "roberta-base"
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel as _RobertaPreTrainedModel
    config = _RobertaPreTrainedModel.config_class.from_pretrained(model_id)
    # config.position_embedding_type = "relative_key_query"
    # print(config)
    model = RobertaForMLM().cuda()
    state_dict = libs_ml.hf_get_state_dict(HF_HOME, model_id, config._commit_hash)
    libs_ml.load_state_dict_with_mapper(model, state_dict, "./.other/roberta_mask_m.txt", "./.other/roberta_mask_s.txt")
    text = "Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols are ignored. This is modified from fairseq's `utils.make_positions`."
    text2 = "Replace non-padding symbols with their position numbers. "
    t = AutoTokenizer.from_pretrained(model_id)
    batch = t([text, text2], padding=True, return_tensors="pt")
    libs_ml.seed_everything(42)
    for i in range(1):
        y: Dict[str, Tensor] = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), output_attn=True)
        print(torch.allclose(y["output"], libs_ml.test_tensor_allclose(idx=2302171555), atol=1e-6))
        print(torch.allclose(y["pool_output"], libs_ml.test_tensor_allclose(idx=2302171556), atol=1e-6))
        print(torch.allclose(y["attn_dist"], torch.stack(libs_ml.test_tensor_allclose(idx=2302171557), 0), atol=1e-6))
        for k, v in y.items():
            print(k, v.shape if isinstance(v, Tensor) else None)
