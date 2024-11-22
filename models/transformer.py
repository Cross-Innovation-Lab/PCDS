from numpy import short
from typing import List
import torch
import torchaudio
import random
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


class Attention(nn.Module):

    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_dim)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask[:, None, None, :].to(torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float32).min
        return extended_attention_mask

    def forward(self, input_from, from_attention_mask, input_to=None, to_attention_mask=None):
        '''
        input_from : seq_num, from_seq_len, dim
        from_attention_mask : seq_num, from_seq_len
        input_to : seq_num, to_seq_len, dim
        to_attention_mask : seq_num, to_seq_len
        '''
        query = self.transpose_for_scores(self.query(input_from))  # [seq_num, num_heads, from_seq_len, head_dim]
        if input_to is not None:
            # cross attention
            attention_mask = self.get_attention_mask(to_attention_mask)
            key = self.transpose_for_scores(self.key(input_to))
            value = self.transpose_for_scores(self.value(input_to))
        else:
            # self attention
            attention_mask = self.get_attention_mask(from_attention_mask)
            key = self.transpose_for_scores(self.key(input_from))
            value = self.transpose_for_scores(self.value(input_from))

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        output_from = torch.matmul(attention_probs, value).permute(0, 2, 1, 3).contiguous().view(input_from.size())
        return output_from
