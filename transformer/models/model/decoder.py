import copy
from torch import nn

class Decoder(nn.Module):

    def __init__(self, decoder_layer, num_layer, norm_layer):
        super().__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(self.num_layer)])
        self.norm_layer = norm_layer


    def forward(self, tgt, encoder_out, tgt_mask, src_tgt_mask):
        out = tgt
        for layer in self.layers:
            out = layer(out, encoder_out, tgt_mask, src_tgt_mask)
            
        out = self.norm_layer(out)

        return out
