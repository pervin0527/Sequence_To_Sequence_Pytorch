import copy

from torch import nn
from models.layer.residual_connection import ResidualConnectionLayer

class DecoderBlock(nn.Module):
    def __init__(self, self_attention, cross_attention, ffn, norm, drop_prob=0):
        super().__init__()
        """
        self_attention : Masked Multi Head Self Attention 객체
        cross_attention : Encoder - Decoder Multi Head Self Attention 객체
        ffn : FeedForward Layer 객체
        norm : LayerNorm 객체
        """
        self.self_attention = self_attention
        self.residual1 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)
        self.ffn = ffn
        self.residual3 = ResidualConnectionLayer(copy.deepcopy(norm), drop_prob)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.ffn)

        return out