from transformers import PreTrainedModel
# from libs import *
from ..._types import *
# from transformers.models.roberta.modeling_roberta import RobertaForMaskedLM

"""
{'_commit_hash': None,
 '_name_or_path': '',
 'architectures': None,
 'bad_words_ids': None,
 'begin_suppress_tokens': None,
 'bos_token_id': None,
 'chunk_size_feed_forward': 0,
 'cross_attention_hidden_size': None,
 'decoder_start_token_id': None,
 'diversity_penalty': 0.0,
 'do_sample': False,
 'early_stopping': False,
 'encoder_no_repeat_ngram_size': 0,
 'eos_token_id': None,
 'exponential_decay_length_penalty': None,
 'finetuning_task': None,
 'forced_bos_token_id': None,
 'forced_eos_token_id': None,
 'id2label': {0: 'LABEL_0', 1: 'LABEL_1'},
 'label2id': {'LABEL_0': 0, 'LABEL_1': 1},
 'length_penalty': 1.0,
 'max_length': 20,
 'min_length': 0,
 'no_repeat_ngram_size': 0,
 'num_beam_groups': 1,
 'num_beams': 1,
 'num_return_sequences': 1,
 'output_scores': False,
 'prefix': None,
 'problem_type': None,
 'remove_invalid_values': False,
 'repetition_penalty': 1.0,
 'return_dict': True,
 'return_dict_in_generate': False,
 'sep_token_id': None,
 'suppress_tokens': None,
 'task_specific_params': None,
 'temperature': 1.0,
 'tf_legacy_loss': False,
 'tokenizer_class': None,
 'top_k': 50,
 'top_p': 1.0,
 'torch_dtype': None,
 'transformers_version': None,
 'typical_p': 1.0,
 'use_bfloat16': False,
}
"""
# from ..._types import *


def create_pos_ids(input_ids: Tensor, padding_idx: int = 1) -> Tensor:
    """input_ids: [N, L]. [2, 3, ..., 1, 1, ...]"""
    mask = input_ids != padding_idx  # choice: m=1
    pos_ids = torch.cumsum(mask, dim=1).mul_(mask).to(dtype=torch.long)
    pos_ids.add_(padding_idx)
    return pos_ids


class RobertaConfig(PretrainedConfig):
    def __init__(
        self,
        #
        add_cross_attention: bool = False,
        add_pooling_layer: bool = False,
        attention_probs_dropout_prob: float = 0.1,
        gradient_checkpointing: bool = False,
        hidden_act: Module = nn.GELU(),
        hidden_dropout_prob: float = 0.1,
        hidden_size: int = 768,
        initializer_range: float = 0.02,
        intermediate_size: int = 3072,
        is_decoder: bool = False,
        layer_norm_eps: float = 1e-5,
        max_position_embeddings: int = 514,
        ninf: float = -1e4,
        num_attention_heads: int = 12,
        num_hidden_layers: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        pad_token_id: int = 1,
        position_embedding_type: Literal["absolute", "relative_key", "relative_key_query"] = "absolute",
        type_vocab_size: int = 1,
        vocab_size: int = 50265,
        #
        tie_word_embeddings: bool = True,
    ) -> None:
        self.model_type: str = "roberta"
        #
        self.add_cross_attention = add_cross_attention
        self.add_pooling_layer = add_pooling_layer
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.gradient_checkpointing = gradient_checkpointing
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.is_decoder = is_decoder
        self.layer_norm_eps = layer_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.ninf = ninf
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.pad_token_id = pad_token_id
        self.position_embedding_type = position_embedding_type
        self.type_vocab_size = type_vocab_size
        self.vocab_size = vocab_size
        # super
        self.is_encoder_decoder = False
        self.tie_encoder_decoder = False
        self.torchscript = False
        self.pruned_heads = set()  # TODO: 需要裁剪的头
        self.tie_word_embeddings = tie_word_embeddings


class RobertaResult:
    def __init__(
        self,
        last_hidden_state: Optional[Tensor] = None,
        pool_output: Optional[Tensor] = None,
        #
        past_key_values: Optional[List[List[Tensor]]] = None,
        hidden_state_list: Optional[List[Tensor]] = None,
        attn_dist_list: Optional[List[Tensor]] = None,
        cross_attn_dist_list: Optional[List[Tensor]] = None,
        #
        mlm_logits: Optional[Tensor] = None,
        mlm_loss: Optional[Tensor] = None,
        mlm_acc: Optional[Tensor] = None,
        #
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.pool_output = pool_output
        self.past_key_values = past_key_values
        self.hidden_state_list = hidden_state_list
        self.attn_dist_list = attn_dist_list
        self.cross_attn_dist_list = cross_attn_dist_list
        self.mlm_logits = mlm_logits
        self.mlm_loss = mlm_loss
        self.mlm_acc = mlm_acc


class RobertaEmbeddings(Module):
    def __init__(
            self,
            config: RobertaConfig
    ) -> None:
        super().__init__()
        self.config = config
        #
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        if config.position_embedding_type == "absolute":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size, config.pad_token_id)
        if config.type_vocab_size > 1:
            self.token_type_embeddings = nn.Embedding(
                config.type_vocab_size+1, config.hidden_size, padding_idx=config.type_vocab_size)
        #
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        input_ids: LongTensor[N, L]
        token_type_ids: LongTensor[N, L]. not const
        return FloatTensor[N, L, E]
        """
        config = self.config
        position_ids = create_pos_ids(input_ids, config.pad_token_id)
        x: Tensor = self.word_embeddings(input_ids)
        if config.position_embedding_type == "absolute":
            x.add_(self.position_embeddings(position_ids))
        if config.type_vocab_size > 1:
            assert isinstance(token_type_ids, Tensor)
            token_type_ids[input_ids == config.pad_token_id] = config.type_vocab_size  # 即padding_idx
            x.add_(self.token_type_embeddings(token_type_ids))
        #
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


# if __name__ == "__main__":
#     config = RobertaConfig(type_vocab_size=2)
#     embedding = RobertaEmbeddings(config)
#     input_ids = torch.tensor([[0, 112, 23, 234, 34, 4564, 2, 3452, 5463, 5675, 2, 1, 1],
#                               [0, 356, 1324, 35, 1234, 444, 234, 1325, 2, 5675, 334, 112, 2]])
#     _0, _1 = torch.zeros(()), torch.ones(())
#     attention_mask = torch.where(input_ids == 1, _0, _1)
#     token_type_ids = torch.tensor([
#         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],

#     ])
#     print(attention_mask)
#     print(embedding(input_ids, token_type_ids))
#     exit()


class RobertaSelfAttention(Module):
    def __init__(
        self,
        config: RobertaConfig
    ) -> None:
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        #
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        #
        E, H = config.hidden_size, config.num_attention_heads
        if config.position_embedding_type in {"relative_key", "relative_key_query"}:
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, E // H)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        encoder_hidden_state: Optional[Tensor] = None,
        past_key_value: Optional[List[Tensor]] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tensor]]]:
        """
        x: hidden_states [N, L, E]
        attention_mask: [N, 1, 1, L], 0,-inf. 
            or encoder_attention_mask
        encoder_hidden_state: [N, Lkv, E]
        past_key_value: Optional[[K:[N, H, L, E//H],V] or [cross_K:[N, H, Lkv, E//H], cross_V]]
        head_mask: [N, H, 1, 1]
        return: output: [N, L, E], attn_dist: [N, H, L, L] or None, 
            past_key_value: Optional[[K:[N, H, L, E//H],V] or [cross_K, cross_V]]
        """
        config = self.config
        E, H = config.hidden_size, config.num_attention_heads
        is_cross_attention = encoder_hidden_state is not None
        if is_cross_attention:
            if past_key_value is not None:
                K = past_key_value[0]
                V = past_key_value[1]
            else:
                K = self._transpose_to_head(self.key(encoder_hidden_state), H)
                V = self._transpose_to_head(self.value(encoder_hidden_state), H)
        else:
            # [N, L, E] -> [N, H, L, E//H]
            K = self._transpose_to_head(self.key(x), H)
            V = self._transpose_to_head(self.value(x), H)
            if past_key_value is not None:
                K = torch.concat([past_key_value[0], K])
                V = torch.concat([past_key_value[1], V])

        if config.is_decoder:  # decoder的第一个, past_key_value=None, 但返回的past_key_value is Tensor
            past_key_value = [K, V]
        Q = self._transpose_to_head(self.query(x), H)  # [N, H, L, E//H]
        attn_scores = Q @ K.transpose(-1, -2)  # [N, H, Lq, Lkv]
        #
        if config.position_embedding_type in {"relative_key", "relative_key_query"} and not is_cross_attention:
            L = x.shape[1]
            position_ids_l = torch.arange(L, dtype=torch.long, device=x.device)  # [:, None]
            position_ids_r = position_ids_l.clone()  # [None, :]
            #
            distance_idx = position_ids_l[:, None] - position_ids_r[None, :] + config.max_position_embeddings - 1
            positional_embedding = self.distance_embedding(distance_idx)  # [L, L, E//H]
            relative_position_scores_Q = torch.einsum("bhld,lrd->bhlr", Q, positional_embedding)
            attn_scores.add_(relative_position_scores_Q)
            if config.position_embedding_type == "relative_key_query":
                relative_position_scores_K = torch.einsum("bhrd,lrd->bhlr", K, positional_embedding)
                attn_scores.add_(relative_position_scores_K)
        #
        attn_scores.div_(math.sqrt(E//H))
        attn_scores.add_(attention_mask)  # [N, 1, 1, Lkv]. for pad
        #
        attn_dist = F.softmax(attn_scores, dim=-1)  # 分布
        attn_dist = self.dropout(attn_dist)  # [N, H, L, L]
        if head_mask is not None:
            attn_dist.mul_(head_mask)
        output: Tensor = attn_dist @ V  # [N, H, Lq, E//H]
        #
        output = self._transpose_from_head(output)  # [N, L, H, E//H] -> [N, L, E]
        if not config.output_attentions:
            attn_dist = None
        return (output, attn_dist, past_key_value)

    @staticmethod
    def _transpose_to_head(x: Tensor, H: int) -> Tensor:
        """
        x: [N, L, E]
        return: [N, H, L, E//H]
        """
        N, L, E = x.shape
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
        new_x_shape = N, L, self.config.hidden_size
        return x.view(new_x_shape)


class RobertaSelfOutput(Module):
    def __init__(self, config: RobertaConfig) -> None:
        """接在self attention后面"""
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        x = self.layer_norm(x + x0)
        return x


class RobertaAttention(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.attn = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        encoder_hidden_state: Optional[Tensor] = None,
        past_key_value: Optional[List[Tensor]] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tensor]]]:
        x0 = x
        x, attn_dist, past_key_value = self.attn(
            x, attention_mask, encoder_hidden_state, past_key_value, head_mask)
        x = self.output(x, x0)
        return x, attn_dist, past_key_value


class RobertaIntermediate(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act_fn = config.hidden_act

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.act_fn(x)
        return x


class RobertaOutput(Module):
    """接在RobertaIntermediate后"""

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        x = self.dense(x)
        x = self.dropout(x)
        x = self.layer_norm(x + x0)
        return x


class RobertaLayer(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = RobertaAttention(config)
        if config.add_cross_attention:
            self.cross_attn = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        encoder_hidden_state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_value: Optional[List[Tensor]] = None,  # [K, V] or [K, V, cross_K, cross_V]
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor], Optional[List[Tensor]]]:
        """
        return: 
            output: [N, L, E], 
            attn_dist: [N, H, L, L]
            cross_attn_dist: [N, H, Lq, Lkv]
            past_key_value: [[N, H, L, E//H], ...]
        """
        config = self.config
        pkv = None
        if past_key_value is not None:
            pkv = past_key_value[:2]
        x, attn_dist, pkv = self.attn(x, attention_mask, None, pkv, head_mask)
        cross_attn_dist = None
        if encoder_hidden_state is not None:
            assert config.add_cross_attention and encoder_attention_mask is not None
            cross_pkv = None
            if past_key_value is not None:
                cross_pkv = past_key_value[2:]
            x, cross_attn_dist, cross_pkv = self.cross_attn(
                x, encoder_attention_mask, encoder_hidden_state, cross_pkv, head_mask)
            if pkv is not None:
                assert cross_pkv is not None
                pkv += cross_pkv
        #
        x0 = x
        x = self.intermediate(x)
        x = self.output(x, x0)
        #
        return x, attn_dist, cross_attn_dist, pkv


class RobertaEncoder(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        x: Tensor,
        attention_mask: Tensor,
        encoder_hidden_state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[List[Tensor]]] = None,
        head_masks: Optional[Tensor] = None,
    ) -> RobertaResult:
        config = self.config
        pkv_list: List[List[Tensor]] = []
        attn_dist_list: Optional[List[Tensor]] = []
        cross_attn_dist_list: Optional[List[Tensor]] = []
        hidden_state_list: Optional[List[Tensor]] = []
        if config.output_hidden_states:
            hidden_state_list.append(x)
        if past_key_values is not None:
            # past_key_values, is_decoder: None, False; None, True; not None True. OK
            assert config.is_decoder
        #
        for i, layer_module in enumerate(self.layer):
            pkv = None
            head_mask = None

            if past_key_values is not None:
                pkv = past_key_values[i]
            if head_masks is not None:
                head_mask = head_masks[i]
            if config.gradient_checkpointing and self.training:
                x, attn_dist, cross_attn_dist, pkv = checkpoint(layer_module, x, attention_mask, encoder_hidden_state,
                                                                encoder_attention_mask, pkv, head_mask, use_reentrant=False)
            else:
                x, attn_dist, cross_attn_dist, pkv = layer_module(x, attention_mask, encoder_hidden_state,
                                                                  encoder_attention_mask, pkv, head_mask)
            #
            if config.is_decoder:
                pkv_list.append(pkv)
            if config.output_attentions:
                attn_dist_list.append(attn_dist)
                if config.add_cross_attention:
                    cross_attn_dist_list.append(cross_attn_dist)
            if config.output_hidden_states:
                hidden_state_list.append(x)

        #
        last_hidden_state = x
        if len(pkv_list) > 0:
            past_key_values = pkv_list
        if len(hidden_state_list) == 0:
            hidden_state_list = None
        if len(attn_dist_list) == 0:
            attn_dist_list = None
        if len(cross_attn_dist_list) == 0:
            cross_attn_dist_list = None
        return RobertaResult(last_hidden_state=last_hidden_state, past_key_values=past_key_values,
                             hidden_state_list=hidden_state_list, attn_dist_list=attn_dist_list,
                             cross_attn_dist_list=cross_attn_dist_list)


class RobertaPool(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, L, E]"""
        x = x[:, 0]
        x = self.dense(x)
        x = x.tanh()
        return x


class RobertaLMHead(Module):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.act_fn = config.hidden_act
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [N, L, E]
        return: [N, L, V]
        """
        x = self.dense(x)
        x = self.act_fn(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class RobertaPreTrainedModel(PreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config: RobertaConfig) -> None:
        self.config: RobertaConfig
        super().__init__(config)

    def _init_weights(self, m: Module) -> None:
        config = self.config
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                m.weight.normal_(mean=0.0, std=config.initializer_range)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Embedding):
                m.weight.normal_(mean=0.0, std=config.initializer_range)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, nn.LayerNorm):
                m.bias.zero_()
                m.weight.fill_(1.0)


class RobertaModel(RobertaPreTrainedModel):
    def __init__(self, config: RobertaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pool = None
        if config.add_pooling_layer:
            self.pool = RobertaPool(config)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        encoder_hidden_state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[List[Tensor]]] = None,
        head_masks: Optional[Tensor] = None,
    ) -> RobertaResult:
        """
        input_ids: LongTensor[N, L]
        attention_mask: LongTensor[N, L]. 1 not mask
        encoder_hidden_state: [N, Lkv, E]
        encoder_attention_mask: [N, Lkv]. 1 not mask
        past_key_values: [n_layers * [[K: [N, H, L, E//H], V] or [K, V, cross_K, cross_V]]]
        head_masks: FloatTensor/LongTensor. [H] or [n_layers, H]
        """
        config = self.config
        if encoder_hidden_state is not None:
            assert encoder_attention_mask is not None
        dtype = self.embeddings.word_embeddings.weight.dtype
        attention_mask = self._create_attn_mask(attention_mask, dtype, config.ninf)
        if head_masks is not None:
            head_masks = self._create_head_masks(head_masks, config.num_hidden_layers, dtype)
        x = self.embeddings(input_ids)
        res: RobertaResult = self.encoder(x, attention_mask, encoder_hidden_state,
                                          encoder_attention_mask, past_key_values, head_masks)
        x = res.last_hidden_state
        if self.pool is not None:
            pool_output = self.pool(x)
            res.pool_output = pool_output
        #
        return res

    @staticmethod
    def _create_attn_mask(attn_mask: Tensor, dtype: Dtype, ninf: float) -> Tensor:
        """
        attn_mask: [N, L], long
        return: [N, 1, 1, L]. float
        """
        attn_mask = attn_mask.to(dtype)
        attn_mask = (1. - attn_mask) * ninf
        return attn_mask[:, None, None, :]

    @staticmethod
    def _create_head_masks(head_masks: Tensor, num_hidden_layers: int, dtype: Dtype) -> Tensor:
        """
        head_masks: [H] or [n_layers, H]. (1 keep, 0 mask)
        return: [n_layers, N, H, 1, 1]
        """
        head_masks = head_masks.to(dtype)
        head_masks = head_masks[..., None, :, None, None]
        if head_masks.ndim == 4:
            head_masks = head_masks[None]
            head_masks = head_masks.broadcast_to((num_hidden_layers, *head_masks.shape[1:]))
        assert head_masks.ndim == 5
        return head_masks

    def get_input_embeddings(self) -> Module:
        return self.embeddings.word_embeddings


class RobertaForMaskedLM(RobertaPreTrainedModel):

    def __init__(self, config: RobertaConfig) -> None:
        super().__init__(config)
        self.config = config
        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.post_init()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Optional[Tensor] = None,
        encoder_hidden_state: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        past_key_values: Optional[List[List[Tensor]]] = None,
        head_masks: Optional[Tensor] = None,
    ) -> RobertaResult:
        """labels: [N, L]"""
        res: RobertaResult = self.roberta(input_ids, attention_mask, encoder_hidden_state,
                                          encoder_attention_mask, past_key_values, head_masks)
        x = res.last_hidden_state
        logits: Tensor = self.lm_head(x)  # 未过softmax

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # ignore_index = -100
            res.mlm_loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
            # acc
            logits = logits.detach()
            masked_id = torch.nonzero(labels != -100, as_tuple=True)
            y_pred = logits[masked_id].argmax(-1)
            y_label = labels[masked_id]
            res.mlm_acc = libs_ml.accuracy(y_pred, y_label, "multiclass", -1)
        else:
            # 在外面计算损失, 可以使用更灵活的损失函数.
            res.mlm_logits = logits
        return res

    def get_output_embeddings(self) -> Module:
        return self.lm_head.decoder


if __name__ == "__main__":
    # from libs import *
    ml.select_device([0])
    model_id = "roberta-base"
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel as _RobertaPreTrainedModel
    config = _RobertaPreTrainedModel.config_class.from_pretrained(model_id)
    # config.position_embedding_type = "relative_key_query"
    # print(config)
    model = RobertaForMaskedLM(RobertaConfig(add_pooling_layer=True,
                               output_attentions=True, gradient_checkpointing=True, output_hidden_states=True)).cuda()
    state_dict = libs_ml.hf_get_state_dict(HF_HOME, model_id, config._commit_hash)
    print(libs_ml.load_state_dict_with_mapper(model, state_dict,
          "./.other/roberta_mask_m.txt", "./.other/roberta_mask_s.txt"))
    text = "Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols are ignored. This is modified from fairseq's `utils.make_positions`."
    text2 = "Replace non-padding symbols with their position numbers. "
    t = AutoTokenizer.from_pretrained(model_id)
    batch = t([text, text2], padding=True, return_tensors="pt")
    ml.seed_everything(42)
    head_masks = torch.randint(0, 2, (12,)).cuda()
    for i in range(1):
        y: RobertaResult = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), head_masks=head_masks)
        print(torch.allclose(y.last_hidden_state, libs_ml.test_tensor_allclose(idx=2302171555), atol=1e-6))
        print(torch.allclose(y.pool_output, libs_ml.test_tensor_allclose(idx=2302171556), atol=1e-6))
        print(torch.allclose(torch.stack(y.attn_dist_list, 0), torch.stack(
            libs_ml.test_tensor_allclose(idx=2302171557), 0), atol=1e-6))
        for k, v in y.__dict__.items():
            print(k, v.shape if isinstance(v, Tensor) else None)



if __name__ == "__main__":
    from libs import *
    ml.select_device([0])
    model_id = "roberta-base"
    from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel as _RobertaPreTrainedModel
    config = _RobertaPreTrainedModel.config_class.from_pretrained(model_id)
    # print(config)
    model = RobertaForMaskedLM(RobertaConfig(add_pooling_layer=True,
                               output_attentions=True, gradient_checkpointing=True, output_hidden_states=True, 
                               position_embedding_type="relative_key_query")).cuda()
    state_dict = libs_ml.hf_get_state_dict(HF_HOME, model_id, config._commit_hash)
    print(libs_ml.load_state_dict_with_mapper(model, state_dict,
          "./.other/roberta_mask_m.txt", "./.other/roberta_mask_s.txt"))
    text = "Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols are ignored. This is modified from fairseq's `utils.make_positions`."
    text2 = "Replace non-padding symbols with their position numbers. "
    t = AutoTokenizer.from_pretrained(model_id)
    batch = t([text, text2], padding=True, return_tensors="pt")
    ml.seed_everything(42)
    head_masks = torch.randint(0, 2, (12,)).cuda()
    for i in range(1):
        y: RobertaResult = model(batch["input_ids"].cuda(), batch["attention_mask"].cuda(), head_masks=head_masks)
        print(torch.allclose(y.last_hidden_state, libs_ml.test_tensor_allclose(idx=2303302352), atol=1e-6))
        print(torch.allclose(y.pool_output, libs_ml.test_tensor_allclose(idx=2303302353), atol=1e-6))
        print(torch.allclose(torch.stack(y.attn_dist_list, 0), torch.stack(
            libs_ml.test_tensor_allclose(idx=2303302354), 0), atol=1e-6))
        for k, v in y.__dict__.items():
            print(k, v.shape if isinstance(v, Tensor) else None)