from torch import nn

class ResidualConnectionLayer(nn.Module):
    def __init__(self, norm, drop_prob=0):
        super().__init__()
        self.norm = norm
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x, sub_layer): ## 입력 텐서와 sublayer를 입력 받는 것에 주목.
        out = x
        out = self.norm(out)
        out = sub_layer(out)
        out = self.dropout(out)
        out = out + x

        # out = x
        # out = sub_layer(out)
        # out = self.dropout(out)
        # out = self.norm(x + out)
        
        return out