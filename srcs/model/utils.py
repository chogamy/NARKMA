import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from transformers.utils import ModelOutput
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import AttentionMaskConverter


@dataclass
class MyModelOutput(ModelOutput):
    """https://github.com/huggingface/transformers/blob/main/src/transformers/utils/generic.py"""

    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class AbsolutePositionEmbedding(nn.Module):
    """
    Reference: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, hidden):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return self.pe[:, : hidden.size(1), :].expand(hidden.size(0), -1, -1)


class Embedding(nn.Module):
    def __init__(
        self,
        emb_size,
        hidden_size,
        pad_token_id,
        max_length,
        pos,
        dropout,
    ) -> None:
        super().__init__()

        self.emb = nn.Embedding(emb_size, hidden_size, padding_idx=pad_token_id)

        if pos == "absolute":
            self.pos = AbsolutePositionEmbedding(
                d_model=hidden_size, max_len=max_length
            )
        else:
            assert 0

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):

        emb = self.emb(input_ids)
        emb = emb + self.pos(emb)

        emb = self.LayerNorm(emb)
        emb = self.dropout(emb)

        return emb


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        activation_funcion="gelu",
        intermediate_size=2048,
        dropout=0.1,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

        self.activation = ACT2FN[activation_funcion]
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden):
        hidden = self.fc1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.fc2(hidden)

        return hidden


class Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper
    BartAttention from https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # cho: remove past key value

        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        # if self.is_decoder:
        #     # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
        #     # Further calls to cross_attention layer can then reuse all cross-attention
        #     # key/value_states (first "if" case)
        #     # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
        #     # all previous decoder key/value_states. Further calls to uni-directional self-attention
        #     # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
        #     # if encoder bi-directional self-attention `past_key_value` is always `None`
        #     past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output
        return attn_output, past_key_value


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size=512,
        num_attention_heads=8,
        activation_funcion="gelu",
        intermediate_size=2048,
        dropout=0.1,
        is_decoder=False,
    ) -> None:
        super().__init__()

        self.self_attn = Attention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            is_decoder=False,
        )
        self.self_attn_LayerNorm = nn.LayerNorm(hidden_size)

        if is_decoder:
            self.cross_attn = Attention(
                embed_dim=hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                is_decoder=is_decoder,
            )

            self.cross_attn_LayerNorm = nn.LayerNorm(hidden_size)
        else:
            self.cross_attn = None

        self.ffn = FeedForward(
            hidden_size=hidden_size,
            activation_funcion=activation_funcion,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.ffn_LayerNorm = nn.LayerNorm(hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden,
        attention_mask,
        cross_hidden_state=None,
        cross_hidden_attention_mask=None,
    ):

        inp = {"hidden_states": hidden, "attention_mask": attention_mask}
        hidden = self.loop(self.self_attn, self.self_attn_LayerNorm, **inp)

        if self.cross_attn is not None:
            inp = {
                "hidden_states": hidden,
                "key_value_states": cross_hidden_state,
                "attention_mask": cross_hidden_attention_mask,
            }
            hidden = self.loop(self.cross_attn, self.cross_attn_LayerNorm, **inp)

        inp = {"hidden": hidden}
        hidden = self.loop(self.ffn, self.ffn_LayerNorm, **inp)

        return hidden

    def loop(self, layer, ln, **kwargs):

        residual = (
            kwargs["hidden"] if "hidden" in kwargs.keys() else kwargs["hidden_states"]
        )

        hidden = layer(**kwargs)

        hidden = self.dropout(hidden)
        hidden = hidden + residual
        hidden = ln(hidden)

        return hidden


class TransformerLayers(nn.Module):
    def __init__(
        self,
        num_hidden_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        activation_funcion="gelu",
        intermediate_size=2048,
        dropout=0.1,
        is_decoder=False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            TransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                activation_funcion=activation_funcion,
                intermediate_size=intermediate_size,
                dropout=dropout,
                is_decoder=is_decoder,
            )
            for _ in range(num_hidden_layers)
        )

    def forward(
        self,
        hidden,
        attention_mask,
        cross_hidden_state=None,
        cross_hidden_attention_mask=None,
    ):
        for idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden,
                attention_mask,
                cross_hidden_state=cross_hidden_state,
                cross_hidden_attention_mask=cross_hidden_attention_mask,
            )

        return hidden


class Transformer(nn.Module):
    def __init__(
        self,
        emb_size=0,
        pos="absolute",
        max_length=256,
        num_hidden_layers=6,
        hidden_size=512,
        num_attention_heads=8,
        activation_funcion="gelu",
        intermediate_size=2048,
        classifier_size=0,
        dropout=0.1,
        is_decoder=False,
        is_causal=False,
    ) -> None:
        super().__init__()

        self.emb = Embedding(
            emb_size=emb_size,
            hidden_size=hidden_size,
            pad_token_id=0,
            max_length=max_length,
            pos=pos,
            dropout=dropout,
        )

        self.blocks = TransformerLayers(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            activation_funcion=activation_funcion,
            intermediate_size=intermediate_size,
            dropout=dropout,
            is_decoder=is_decoder,
        )

        self.classifier = nn.Linear(hidden_size, classifier_size)

        self.self_attn_mask_converter = AttentionMaskConverter(is_causal=False)
        self.cross_attn_mask_converter = AttentionMaskConverter(is_causal=is_causal)

    def forward(
        self,
        input_ids,
        attention_mask,
        cross_hidden_state=None,
        cross_hidden_attention_mask=None,
        output_last_hidden_state=False,
        output_logits=False,
    ):

        hidden = self.emb(input_ids)

        attention_mask = self.self_attn_mask_converter.to_4d(
            attention_mask, hidden.size(1), hidden.dtype
        )
        if cross_hidden_attention_mask is not None:
            cross_hidden_attention_mask = self.cross_attn_mask_converter.to_4d(
                cross_hidden_attention_mask,
                cross_hidden_attention_mask.size(1),
                hidden.dtype,
                hidden.size(1),
            )

        hidden = self.blocks(
            hidden, attention_mask, cross_hidden_state, cross_hidden_attention_mask
        )
        logits = self.classifier(hidden)

        return MyModelOutput(
            logits=logits if output_logits else None,
            last_hidden_state=hidden if output_last_hidden_state else None,
        )
