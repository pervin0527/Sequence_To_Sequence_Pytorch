import copy
from torch import nn

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layer, norm_layer):
        super().__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_layer)])
        self.norm_layer = norm_layer


    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)

        out = self.norm_layer(out)

        return out
