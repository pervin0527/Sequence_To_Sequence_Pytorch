import copy

from torch import nn
from models.layer.residual_connection import ResidualConnection

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, position_ff, norm_layer, dr_rate=0):
        super().__init__()
        self.self_attention = self_attention
        self.residual1 = ResidualConnection(copy.deepcopy(norm_layer), dr_rate)
        
        self.cross_attention = cross_attention
        self.residual2 = ResidualConnection(copy.deepcopy(norm_layer), dr_rate)
        
        self.position_ff = position_ff
        self.residual3 = ResidualConnection(copy.deepcopy(norm_layer), dr_rate)

    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        out = self.residual1(out, lambda out: self.self_attention(query=out, key=out, value=out, mask=tgt_mask))
        out = self.residual2(out, lambda out: self.cross_attention(query=out, key=encoder_out, value=encoder_out, mask=src_tgt_mask))
        out = self.residual3(out, self.position_ff)
        
        return out
