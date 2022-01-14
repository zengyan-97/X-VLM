# Copyright 2021 The OpenAI Team Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import logging


logger = logging.get_logger(__name__)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.T)
    return (caption_loss + image_loss) / 2.0


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size



    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class CLIPMLP(nn.Module):
    def __init__(self, hidden_act, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = ACT2FN[hidden_act]
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size):
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_attention_heads, attention_dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.mlp = CLIPMLP(hidden_act, hidden_size, intermediate_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: None,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(batch, seq_len, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=None,
            output_attentions=False,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of :obj:`config.num_hidden_layers` self attention layers. Each layer is a
    :class:`~transformers.CLIPEncoderLayer`.

    Args:
        config: CLIPConfig
    """

    def __init__(self, hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size, num_hidden_layers, local_attn_depth):
        super().__init__()
        self.depth = num_hidden_layers
        self.local_attn_depth = local_attn_depth
        self.layers = nn.ModuleList([CLIPEncoderLayer(hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size) for _ in range(num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        idx_to_group_img=None,
        image_atts=None
    ):
        r"""
        Args:
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            causal_attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Causal mask for the text model. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """

        do_gather = True if idx_to_group_img is not None else False

        if do_gather and (image_atts is not None):
            full_atts = torch.ones(inputs_embeds.shape[:2], dtype=inputs_embeds.dtype).to(inputs_embeds.device)
            image_atts_blk = torch.cat([image_atts, full_atts], dim=0)

            image_atts_blk = image_atts_blk.unsqueeze(1).unsqueeze(2)
            image_atts_blk = (1.0 - image_atts_blk) * -10000.0
            # (bs, 1, 1, num_patches)
            image_atts_blk = image_atts_blk.expand(-1, -1, image_atts_blk.size(-1), -1)
        else:
            image_atts_blk = None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if (self.local_attn_depth > 0) and (idx >= self.depth - self.local_attn_depth):
                if do_gather:
                    do_gather = False
                    hidden_states_bs = torch.gather(hidden_states, dim=0, index=idx_to_group_img.view(-1, 1, 1).expand(-1, hidden_states.shape[1], hidden_states.shape[2]))
                    hidden_states = torch.cat([hidden_states_bs, hidden_states], dim=0)

                hidden_states = encoder_layer(hidden_states, attention_mask=image_atts_blk)
            else:
                hidden_states = encoder_layer(hidden_states, attention_mask=None)

        return hidden_states


class CLIPVisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size, num_hidden_layers, local_attn_depth=0):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patch_embed = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            in_channels=3, out_channels=hidden_size, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.class_embedding = nn.Parameter(torch.randn(hidden_size))
        self.num_pos_embed = self.num_patch_embed + 1
        self.pos_embed = nn.Embedding(self.num_pos_embed, hidden_size)
        self.register_buffer("position_ids", torch.arange(self.num_pos_embed).expand((1, -1)))

        self.pre_layrnorm = nn.LayerNorm(hidden_size)
        self.encoder = CLIPEncoder(hidden_size, hidden_act, num_attention_heads, attention_dropout, intermediate_size,
                                   num_hidden_layers, local_attn_depth=local_attn_depth)
        self.post_layernorm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x,
        idx_to_group_img=None,
        image_atts=None
    ):

        batch_size = x.shape[0]
        patch_embeds = self.patch_embed(x)  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        hidden_states = embeddings + self.pos_embed(self.position_ids)

        hidden_states = self.pre_layrnorm(hidden_states)

        outputs = self.encoder(
            inputs_embeds=hidden_states,
            idx_to_group_img=idx_to_group_img,
            image_atts=image_atts)

        outputs = self.post_layernorm(outputs)

        if idx_to_group_img is not None:
            bs = len(idx_to_group_img)
            outputs, outputs_fullatts = torch.split(outputs, [bs, outputs.size(0)-bs])
            return outputs, outputs_fullatts

        return outputs
