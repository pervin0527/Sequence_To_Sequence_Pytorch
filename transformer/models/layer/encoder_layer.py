import copy

from torch import nn
from models.layer.residual_connection import ResidualConnection

class EncoderLayer(nn.Module):
    def __init__(self, self_attention, position_ff, norm_layer, dr_rate=0):
        super().__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnection(copy.deepcopy(norm_layer), dr_rate)

        self.position_ff = position_ff
        self.residual2 = ResidualConnection(copy.deepcopy(norm_layer), dr_rate)


    def forward(self, src, src_mask):
        ## src : [batch_size, max_seq_len, d_model]
        out = src

        ## Self-Attention이므로 query, key, value가 모두 out으로 동일.
        ## 이 때, lambda가 적용되는 이유는 순환의 목적이 아니라 reisdual 객체 내부에서 필요한 시점에 호출하기 위함.
        out = self.residual1(out, lambda out : self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.position_ff)

        return out