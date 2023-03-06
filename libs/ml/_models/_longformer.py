
"""
cls_token_id=0, pad_token_id=1
"""

from ..._types import *
from .._ml_alg._metrics import accuracy
from ._roberta import (RobertaEmbeddings, RobertaAttention, RobertaIntermediate, RobertaOutput,
                        RobertaPool, RobertaModel, RobertaLayer, RobertaSelfAttention,
                        RobertaSelfOutput, RobertaLMHead, RobertaPreTrainedModel)


class LongformerEmbeddings(RobertaEmbeddings):
    def __init__(self,
                 vocab_size: int = 50265,
                 hidden_size: int = 768,
                 max_pos_embeddings: int = 4098,
                 pad_token_id: int = 1,
                 layer_norm_eps: float = 1e-5,
                 dropout_p: float = 0.1) -> None:
        super().__init__(vocab_size, hidden_size, max_pos_embeddings, pad_token_id, layer_norm_eps, dropout_p)


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
#     embedding = LongformerEmbeddings()
#     res = embedding(input_ids)
#     print(res)


class LongformerSelfAttention(RobertaSelfAttention):
    def __init__(self, layer_id: int,
                 hidden_size: int = 768, n_attn_heads: int = 12,
                 attn_dropout_p: float = 0.1,
                 attn_window: Union[int, List[int]] = 512):
        super().__init__(hidden_size, n_attn_heads, attn_dropout_p)
        self.query_global = nn.Linear(hidden_size, hidden_size)
        self.key_global = nn.Linear(hidden_size, hidden_size)
        self.value_global = nn.Linear(hidden_size, hidden_size)
        self.layer_id = layer_id
        #
        if isinstance(attn_window, int):
            attn_window_i = attn_window
        else:
            attn_window_i = attn_window[layer_id]
        assert attn_window_i % 2 == 0
        self.attn_window_side = attn_window_i // 2

    def _get_attn_dist(
        self,
        hidden_states: Tensor,  # [N, L, E]
        attn_mask: Tensor,  # [N, L]
    ) -> Tensor:
        E, H = self.hidden_size, self.n_attn_heads
        Q = self._transpose_to_head(self.query(hidden_states))  # [N,H,L,E//H]
        K = self._transpose_to_head(self.key(hidden_states))
        Q.div_(math.sqrt(E//H))
        attn_scores = self._get_window_attn_scores(Q, K, self.attn_window_side)  # [N,H,L,w+1]
        # 请对k0进行mask.
        attn_mask = attn_mask[:, None, :,  None]  # [N,H,L,E//H]=[N,1,L,1]
        # diagonal_mask类似于roberta中: 对所有的Q, 执行相同mask_K.
        diagonal_mask = self._get_window_attn_scores(torch.ones_like(attn_mask), attn_mask,
                                                     self.attn_window_side)  # [N,H,L,w+1]
        attn_scores += diagonal_mask
        # k0 -> all q
        K0 = K[:, :, 0:1]  # [N,H,1,E//H]
        k0_global_attn_scores = Q @ K0.transpose(-1, -2)  # [N,H,L,1]
        attn_scores = torch.concat([k0_global_attn_scores, attn_scores], -1)  # [N,H,L,1+w+1]
        attn_dist = F.softmax(attn_scores, dim=-1)
        # 对所有的K, 执行相同的mask_Q
        # attn_dist = attn_dist.masked_fill_(attn_mask != 0, 0.)
        attn_dist = self.dropout(attn_dist)
        return attn_dist

    def forward(
        self,
        hidden_states: Tensor,  # [N, L, E]
        attn_mask: Tensor,  # NINF: mask. [N, L]
        output_attn: bool = False,
    ) -> Tuple[Tensor, ...]:
        """(规定cls_token=0, 为global_attn_token: 方便实现)
        hidden_states: [N, L(%w==0), E]
        return:
            output: [N,L,E]
            attn_dist: [N,H,L,1+w+1]. {attn_dist[0]为k0_global, 没有对L(Q,上层)进行mask-同roberta}
            global_attn_dist: [N,H,1,L]
        """
        #
        attn_dist = self._get_attn_dist(hidden_states, attn_mask)
        #
        V = self._transpose_to_head(self.value(hidden_states))
        output = self._compute_output(attn_dist, V)  # [N,L,E]
        #
        q0_output, q0_attn_dist = self._compute_global_q0_output(hidden_states, attn_mask)  # [N, 1, E], [N,H,1,L]
        #
        output[:, 0] = q0_output[:, 0]
        return (output,) if not output_attn else (output, attn_dist, q0_attn_dist)

    @staticmethod
    def _diag_to_col(x: Tensor) -> Tensor:
        """
        x: [N, H, C, w, w]
        return: [N, H, C, w, w+1]
        """
        N, H, C, w, _ = x.shape
        x = F.pad(x, (0, 0, 0, 1))  # [N, H, c, w+1, w]
        return x.view(N, H, C, w, w+1)  # 将对角线拉成列

    @staticmethod
    def _pad_and_diagonalize(attn_dist: Tensor) -> Tensor:
        """
        attn_dist: [N, H, L//w1, w1, w+1]
        return: [N, H, L//w1, w1, 3*w1]
        """
        N, H, Ldw1, w1, _ = attn_dist.shape
        attn_dist = F.pad(attn_dist, (0, w1+1))  # [N*H, L//w1, w1, w+1+w1+1]
        attn_dist = attn_dist.view(N, H, Ldw1, -1)
        attn_dist = attn_dist[:, :, :, :-w1]
        attn_dist = attn_dist.view(N, H, Ldw1, w1, 3*w1+1)
        attn_dist = attn_dist[:, :, :, :, :-1]
        return attn_dist

    @staticmethod
    def _chunk(x: Tensor, w1: int = 256) -> Tensor:
        """
        x: [N, H, L, E//H]
        return: [N, H, C, w, E//H]
        """
        N, H, L, EdH = x.shape
        w = w1 * 2
        n_chunks = L//w*2-1  # C
        _size = (N, H, n_chunks, w, EdH)  # [N, H, C, w, E//H]
        _stride = x.stride()
        _stride = [*_stride[:2],  _stride[2]*w1, *_stride[2:]]  # [s0, s1, s2*w1, s2, s3]
        return x.as_strided(size=_size, stride=_stride)

    @staticmethod
    def _mask_invalid_locations(x: Tensor, w1: int) -> None:
        """
        x: [N, H, L, w+1]
        """
        NINF = torch.finfo(x.dtype).min
        beginning_mask_2d = x.new_ones(w1, w1 + 1).tril().flip(dims=[0])  # [w1, w1+1]
        beginning_mask = beginning_mask_2d[None, None]
        ending_mask = beginning_mask.flip(dims=(2, 3))
        beginning_input = x[:, :, :w1, : w1 + 1]
        beginning_mask = beginning_mask.expand(beginning_input.shape)
        x[:, :, :w1, : w1 + 1] = torch.full_like(beginning_input, NINF).where(beginning_mask.bool(), beginning_input)
        ending_input = x[:, :, -w1:, -(w1 + 1):]
        ending_mask = ending_mask.expand(ending_input.shape)
        x[:, :, -w1:, -(w1 + 1):] = torch.full_like(ending_input, NINF).where(ending_mask.bool(), ending_input)

    def _get_window_attn_scores(self, Q: Tensor, K: Tensor, w1: int = 256) -> Tensor:
        """
        Q: [N, H, L, E//H]
        K: [N, H, L, E//H]
        return: [N,H,L,w+1]
        """
        # attn_window_side==window_overlap
        # 令w=window_size, w1=w//2, 表示one side or overlap.
        dtype, device = Q.dtype, Q.device
        N, H, L, _ = Q.shape
        w = 2 * w1
        #
        Q = self._chunk(Q, w1)  # [N, H, C, w, E//H]
        K = self._chunk(K, w1)
        attn_scores = Q @ K.transpose(-1, -2)  # [N, H, C, w, w]
        #
        attn_scores = self._diag_to_col(attn_scores)  # [N, H, C, w, w+1]
        diag_attn_scores = torch.zeros((N, H, L//w1, w1, w+1), dtype=dtype, device=device)
        diag_attn_scores[:, :, :-1, :, w1:] = attn_scores[:, :, :, :w1, : w1 + 1]
        diag_attn_scores[:, :, -1, :, w1:] = attn_scores[:, :, -1, w1:, : w1 + 1]
        diag_attn_scores[:, :, 1:, :, :w1] = attn_scores[:, :, :, -(w1 + 1): -1, w1 + 1:]
        diag_attn_scores[:, :, 0, 1:w1, 1:w1] = attn_scores[:, :, 0, : w1 - 1, 1 - w1:]
        #
        diag_attn_scores = diag_attn_scores.view(N, H, L, w+1)
        self._mask_invalid_locations(diag_attn_scores, w1)
        return diag_attn_scores

    def _get_window_output(
        self, attn_dist: Tensor, V: Tensor, w1: int
    ) -> Tensor:
        """
        attn_dist: [N, H, L, w+1]
        V: [N, H, L, E//H]
        """

        N, H, L, _ = V.shape
        E = self.hidden_size
        w = 2 * w1
        attn_dist = attn_dist.view(N, H, L//w1, w1, w+1)
        V = F.pad(V, (0, 0, w1, w1), value=-1)  # [N,H,w1+L+w1,E//H]
        _size = (N, H, L//w1, 3*w1, E//H)
        _stride = V.stride()
        _stride = [*_stride[:2], _stride[2]*w1, *_stride[2:]]
        V = V.as_strided(size=_size, stride=_stride)
        attn_dist = self._pad_and_diagonalize(attn_dist)  # [N, H, L//w1, w1, 3*w1]
        output = attn_dist @ V  # [N,H,L//w1,w1,E//H]
        return output.view(N, H, L, E//H)

    def _compute_output(
        self,
        attn_dist: Tensor,  # [N, H, L, 1+w+1]
        V: Tensor,  # [N,L,E]
    ) -> Tensor:
        attn_dist0 = attn_dist[:, :, :, 0:1]  # [N,H,L,1]
        V0 = V[:, :, 0:1, :]  # [N,H,1,E//H]
        attn_output = attn_dist0 @ V0  # [N,H,L,E//H]
        #
        attn_dist = attn_dist[:, :, :, 1:]  # [N.H,L,w+1]
        attn_output_win = self._get_window_output(attn_dist, V, self.attn_window_side)  # [N,H,L,E//H]
        attn_output.add_(attn_output_win)
        return self._transpose_from_head(attn_output)

    def _compute_global_q0_output(
        self,
        hidden_states: Tensor,
        attn_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        hidden_states: [N, L, E]
        attn_mask: [N, L]
        return: g_output: [N,1,E], g_attn_dist: [N,H,1,L]
        """
        _, _, E = hidden_states.shape
        H = self.n_attn_heads
        GQ0 = self._transpose_to_head(self.query_global(hidden_states[:, 0:1, :]))  # [N, H, 1, E//H]
        GK = self._transpose_to_head(self.key_global(hidden_states))
        GV = self._transpose_to_head(self.value_global(hidden_states))
        GQ0.div_(math.sqrt(E//H))
        q0_attn_scores = GQ0 @ GK.transpose(-1, -2)  # [N,H,1,L]
        attn_mask = attn_mask.clone()
        attn_mask[:, 0] = 0
        q0_attn_scores += attn_mask[:, None, None]  # 对q0的进行mask_K
        q0_attn_dist = F.softmax(q0_attn_scores, dim=-1)
        q0_attn_dist = self.dropout(q0_attn_dist)

        # global attn output
        global_output = q0_attn_dist @ GV  # [N, H, 1, E//H]
        global_output = self._transpose_from_head(global_output)  # [N, 1, E]
        return global_output, q0_attn_dist


# if __name__ == "__main__":
#     ml.seed_everything(42)
#     attn = LongformerSelfAttention(0, attn_window=[256])
#     ml.seed_everything(42)
#     x = torch.randn(2, 512, 768)
#     attention_mask = libs_ml.test_tensor_allclose(idx=2302171828)
#     attention_mask[:, 0] = attention_mask[0, -1]
#     # attention_mask = (attention_mask != 0)
#     # attention_mask = attention_mask.to(torch.float32).masked_fill(attention_mask, -10000.0)
#     y = attn.forward(x, attention_mask)


class LongformerSelfOutput(RobertaSelfOutput):
    def __init__(self, hidden_size: int = 768, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super().__init__(hidden_size, layer_norm_eps, dropout_p)


class LongformerAttention(RobertaAttention):
    def __init__(self, layer_id: int, hidden_size: int = 768,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1, attn_window: Union[int, List[int]] = 512,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super(RobertaAttention, self).__init__()
        self.attn = LongformerSelfAttention(layer_id, hidden_size, n_attn_heads, attn_dropout_p, attn_window)
        self.output = LongformerSelfOutput(hidden_size, layer_norm_eps, dropout_p)


class LongformerIntermediate(RobertaIntermediate):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072) -> None:
        super().__init__(hidden_size, intermediate_size)


class LongformerOutput(RobertaOutput):
    def __init__(self, hidden_size: int = 768, intermediate_size: int = 3072,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super().__init__(hidden_size, intermediate_size, layer_norm_eps, dropout_p)


class LongformerLayer(RobertaLayer):
    def __init__(self, layer_id: int,  hidden_size: int = 768, intermediate_size: int = 3072, n_attn_heads: int = 12,
                 attn_dropout_p: float = 0.1, attn_window: Union[int, List[int]] = 512,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1) -> None:
        super(RobertaLayer, self).__init__()
        self.attn = LongformerAttention(layer_id, hidden_size, n_attn_heads,
                                        attn_dropout_p, attn_window, layer_norm_eps, dropout_p)
        self.intermediate = LongformerIntermediate(hidden_size, intermediate_size)
        self.output = LongformerOutput(hidden_size, intermediate_size, layer_norm_eps, dropout_p)


class LongformerEncoder(Module):
    def __init__(self, n_layers: int = 12,
                 hidden_size: int = 768, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1,
                 attn_window: Union[int, List[int]] = 512,
                 layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 gradient_checkpoint: bool = False) -> None:
        super().__init__()
        self.gradient_checkpoint = gradient_checkpoint
        self.layer = nn.ModuleList([LongformerLayer(i, hidden_size, intermediate_size, n_attn_heads, attn_dropout_p, attn_window,
                                                    layer_norm_eps, dropout_p) for i in range(n_layers)])

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        attn_list: List[Tensor] = []
        global_attn_list: List[Tensor] = []
        for _, layer_module in enumerate(self.layer):
            if self.gradient_checkpoint:
                x_tuple = checkpoint(layer_module, x, attn_mask, output_attn)
            else:
                x_tuple = layer_module(x, attn_mask, output_attn)
            #
            if output_attn:
                attn_list.append(x_tuple[1])
                global_attn_list.append(x_tuple[2])
            x = x_tuple[0]
        res = {"output": x}
        if output_attn:
            res["attn_dist"] = torch.stack(attn_list, dim=0)
            res["global_attn_dist"] = torch.stack(global_attn_list, dim=0)
        return res


class LongformerPool(RobertaPool):
    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__(hidden_size)


class LongformerLMHead(RobertaLMHead):
    def __init__(self, hidden_size: int = 768, layer_norm_eps: float = 1e-5, vocab_size: int = 50265) -> None:
        super().__init__(hidden_size, layer_norm_eps, vocab_size)


LongformerPreTrainedModel = RobertaPreTrainedModel


class LongformerModel(LongformerPreTrainedModel):
    def __init__(self, vocab_size: int = 50265, hidden_size: int = 768, max_pos_embeddings: int = 4098,
                 pad_token_id: int = 1, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 n_layers: int = 12, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1, attn_window: Union[int, List[int]] = 512,
                 add_pooling_layer: bool = True, init_range: Optional[float] = 0.02,
                 gradient_checkpoint: bool = False) -> None:
        super().__init__(init_range)
        self.max_attn_window = attn_window if isinstance(attn_window, int) else max(attn_window)
        self.pad_token_id = pad_token_id
        self.embeddings = LongformerEmbeddings(
            vocab_size, hidden_size, max_pos_embeddings, pad_token_id, layer_norm_eps, dropout_p)
        self.encoder = LongformerEncoder(n_layers, hidden_size, intermediate_size,
                                         n_attn_heads, attn_dropout_p, attn_window, layer_norm_eps, dropout_p, gradient_checkpoint)
        self.pool = None
        if add_pooling_layer:
            self.pool = LongformerPool(hidden_size)
        self.post_init()

    @staticmethod
    def _pad_to_window_size(
        input_ids: Tensor,  # [N, L0]
        attn_mask: Tensor,  # [N, L0]
        pad_token_id: int,
        max_attn_window: int,
    ) -> Tuple[int, Tensor, Tensor]:
        """
        return: P, input_ids: [N, L], attn_mask: [N, L]
        """
        L = input_ids.shape[1]
        P = (max_attn_window - L % max_attn_window) % max_attn_window
        if P > 0:
            input_ids = F.pad(input_ids, (0, P), value=pad_token_id)
            attn_mask = F.pad(attn_mask, (0, P), value=False)

        return P, input_ids, attn_mask

    def _unpad_result(self, result: Dict[str, Tensor], P: int) -> None:
        for k, v in result.items():
            if k == "output":
                result[k] = v[:, :-P]
            elif k == "attn_dist":
                result[k] = v[:, :, :, :-P]
            elif k == "global_attn_dist":
                result[k] = v[:, :, :, :, :-P]

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,  # 1:表示 attn
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        """
        input_ids: [N, L]
        attn_mask: [N, L]
        return: output: [N, L, E], pool_output: [N, E], attn_dist: [NL, N, H, L, L]
        """
        P, input_ids, attn_mask = self._pad_to_window_size(input_ids, attn_mask, self.pad_token_id, self.max_attn_window)
        attn_mask[:, 0] = 0
        #
        res = {}
        dtype = self.embeddings.word_embeddings.weight.dtype
        attn_mask = RobertaModel._create_attn_mask(attn_mask, dtype)
        x = self.embeddings(input_ids)
        res = self.encoder(x, attn_mask, output_attn)
        x = res["output"]
        if self.pool is not None:
            pool_output = self.pool(x)
            res["pool_output"] = pool_output
        #
        self._unpad_result(res, P)
        return res


class LongformerForMLM(LongformerPreTrainedModel):
    def __init__(self, vocab_size: int = 50265, hidden_size: int = 768, max_pos_embeddings: int = 4098,
                 pad_token_id: int = 1, layer_norm_eps: float = 1e-5, dropout_p: float = 0.1,
                 n_layers: int = 12, intermediate_size: int = 3072,
                 n_attn_heads: int = 12, attn_dropout_p: float = 0.1, attn_window: Union[int, List[int]] = 512,
                 add_pooling_layer: bool = True, init_range: Optional[float] = 0.02,
                 gradient_checkpoint: bool = False) -> None:
        super().__init__(init_range, "longformer")
        self.vocab_size = vocab_size
        self.longformer = LongformerModel(vocab_size, hidden_size, max_pos_embeddings, pad_token_id, layer_norm_eps, dropout_p,
                                          n_layers, intermediate_size, n_attn_heads, attn_dropout_p, attn_window, add_pooling_layer, None, gradient_checkpoint)
        self.lm_head = LongformerLMHead(hidden_size, layer_norm_eps, vocab_size)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attn_mask: Tensor,
        labels: Optional[Tensor] = None,
        output_attn: bool = False,
    ) -> Dict[str, Tensor]:
        """return: {output, pool_output; mlm_loss, mlm_acc, attn_dist, global_attn_dist}"""
        res = self.longformer(input_ids, attn_mask, output_attn)
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
            res["mlm_acc"] = accuracy(y_pred, y_label, "multiclass", -1)
        return res


if __name__ == "__main__":
    from libs import *
    ml.select_device([0])
    from transformers.models.longformer.modeling_longformer import LongformerPreTrainedModel as _LongformerPreTrainedModel
    model_id = "allenai/longformer-base-4096"
    config = _LongformerPreTrainedModel.config_class.from_pretrained(model_id)
    # config.position_embedding_type = "relative_key_query"
    # print(config)
    model = LongformerForMLM()
    model.cuda()
    state_dict = libs_ml.hf_get_state_dict(HF_HOME, model_id, config._commit_hash)
    libs_ml.load_state_dict_with_mapper(
        model, state_dict, "./.other/longformer_mask_m.txt", "./.other/longformer_mask_s.txt")
    text = "Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols are ignored. This is modified from fairseq's `utils.make_positions`."
    text2 = "Replace non-padding symbols with their position numbers. "
    t = AutoTokenizer.from_pretrained(model_id)
    batch = t([text, text2], padding=True, return_tensors="pt")
    ml.seed_everything(42, True)
    for i in range(1):
        y: Dict[str, Tensor] = ml.test_time(lambda: model(
            batch["input_ids"].cuda(), batch["attention_mask"].cuda(), output_attn=True))
        ########
        # print(torch.allclose(y["output"], libs_ml.test_tensor_allclose(idx=2302172126), atol=1e-5))
        # print(torch.allclose(y["pool_output"], libs_ml.test_tensor_allclose(idx=2302172127), atol=1e-6))
        # print(torch.allclose(y["attn_dist"], torch.stack(
        #     libs_ml.test_tensor_allclose(idx=2302172128), 0), atol=1e-6))
        # # my: torch.Size([2, 12, 1, 512])
        # gad = torch.stack(libs_ml.test_tensor_allclose(idx=2302172129), 0).transpose(-1, -2)
        # print(torch.allclose(y["global_attn_dist"], gad[:, :, :, :, :45], atol=1e-6))
        # print(torch.allclose(torch.zeros_like(gad[:, :, :, :, 45:]), gad[:, :, :, :, 45:], atol=1e-6))
        ########## -mask, -transpose in dropout
        # libs_ml.test_tensor_allclose(y["output"], idx=2302180148)
        # libs_ml.test_tensor_allclose(y["pool_output"], idx=2302180149)
        # libs_ml.test_tensor_allclose(y["attn_dist"], idx=2302180150)
        # libs_ml.test_tensor_allclose(y["global_attn_dist"], idx=2302180151)
        #
        output = libs_ml.test_tensor_allclose(idx=2302180148)
        pool_output = libs_ml.test_tensor_allclose(idx=2302180149)
        attn_dist = libs_ml.test_tensor_allclose(idx=2302180150)
        global_attn_dist = libs_ml.test_tensor_allclose(idx=2302180151)
        print(torch.allclose(y["output"], output))
        print(torch.allclose(y["pool_output"], pool_output))
        print(torch.allclose(y["attn_dist"], attn_dist))
        print(torch.allclose(y["global_attn_dist"], global_attn_dist))
        for k, v in y.items():
            print(k, v.shape if isinstance(v, Tensor) else None)
