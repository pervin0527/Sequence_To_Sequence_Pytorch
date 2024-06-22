import copy
from torch import nn

class Encoder(nn.Module):
    def __init__(self, encoder_block, num_layer, norm):
        super().__init__()
        self.num_layer = num_layer
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(self.num_layer)])
        self.norm = norm

    def forward(self, src, src_mask):
        out = src
        for layer in self.layers:
            out = layer(out, src_mask)
            
        out = self.norm(out)

        return out
