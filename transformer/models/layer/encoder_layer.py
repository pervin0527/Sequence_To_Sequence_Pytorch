import copy

from torch import nn
from models.layer.residual_connection import ResidualConnectionLayer

class EncoderBlock(nn.Module):
    def __init__(self, self_attention, ffn, norm, drop_prob=0):
        """
        self_attention : Self-Attention Layer 객체
        ffn : FeedForward Layer 객체
        norm : LayerNorm 겍체
        """
        super().__init__()
        self.self_attention = self_attention
        ## 여기서 copy.deepcopy()는 하나의 layer 객체를 서로 다른 객체로 복사하기 위함.
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)

        self.ffn = ffn
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)

    def forward(self, src, src_mask):
        ## src : [batch_size, max_seq_len, d_model]
        out = src

        ## Self-Attention이므로 query, key, value가 모두 out으로 동일.
        ## 이 때, lambda가 적용되는 이유는 순환의 목적이 아니라 reisdual 객체 내부에서 필요한 시점에 호출하기 위함.
        out = self.residual1(out, lambda out : self.self_attention(query=out, key=out, value=out, mask=src_mask))
        out = self.residual2(out, self.ffn)

        return out