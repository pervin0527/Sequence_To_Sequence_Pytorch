import copy
import math
import torch

from torch import nn
from torch.nn import functional as F

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob=0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        ## Embedded vector를 Q, K, V 행렬로 변환하기 위한 Layers
        self.q_fc = nn.Linear(d_model, d_model) # (d_model, d_model)
        self.k_fc = nn.Linear(d_model, d_model) # (d_model, d_model)
        self.v_fc = nn.Linear(d_model, d_model) # (d_model, d_model)

        ## output layer
        self.out_fc = nn.Linear(d_model, d_model) # (d_model, d_model)

        self.dropout = nn.Dropout(p=drop_prob)


    def calculate_attention(self, query, key, value, mask):
        ## Q, K, V: (batch_size, num_heads, seq_len, d_k)
        ## Padding Mask : (batch_size, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Attention Score. Q x K^T, (n_batch, num_heads, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k) ## Scaling

        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
            
        attention_prob = F.softmax(attention_score, dim=-1) ## Attention Distribution. (n_batch, num_heads, seq_len, seq_len)
        attention_prob = self.dropout(attention_prob)
        out = torch.matmul(attention_prob, value) ## Attention Value. (n_batch, num_heads, seq_len, d_k)

        return out


    def forward(self, query, key, value, mask=None):
        ## (query, key, value)는 동일한 input tensor : (n_batch, seq_len, d_model)
        ## transform(x, fc)가 Q, K, V로 변환.
        ## mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, num_heads, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc): # (n_batch, seq_len, d_model)
            out = fc(x)       # (n_batch, seq_len, d_model)

            # (n_batch, seq_len, num_heads, d_k)
            out = out.view(n_batch, -1, self.num_heads, self.d_model // self.num_heads)
            out = out.transpose(1, 2) # (n_batch, num_heads, seq_len, d_k)

            return out

        query = transform(query, self.q_fc) # (n_batch, num_heads, seq_len, d_k)
        key = transform(key, self.k_fc)       # (n_batch, num_heads, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, num_heads, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, num_heads, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, num_heads, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_model)

        return out
