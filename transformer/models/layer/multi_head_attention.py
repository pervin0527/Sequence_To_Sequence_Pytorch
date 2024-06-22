import copy
import math
import torch

from torch import nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, qkv_fc, out_fc, dr_rate=0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.o_fc = out_fc

        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc): # (n_batch, seq_len, d_embed)
            out = fc(x)       # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.num_heads, self.d_model//self.num_heads) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)       # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)

        out = self.o_fc(out) # (n_batch, seq_len, d_embed)

        return out

    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)

        attention_dist = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        attention_dist = self.dropout(attention_dist)

        attention_value = torch.matmul(attention_dist, value) # (n_batch, h, seq_len, d_k)

        return attention_value