from torch import nn

class ResidualConnection(nn.Module):
    def __init__(self, norm_layer, dr_rate=0):
        super().__init__()
        self.norm_layer = norm_layer
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x, sub_layer):
        out = x
        out = self.norm_layer(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x

        return x