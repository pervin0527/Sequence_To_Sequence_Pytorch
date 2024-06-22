from torch import nn

class TransformerEmbedding(nn.Module):
    def __init__(self, token_embedder, position_encoder, dr_rate=0):
        super().__init__()
        self.embedding = nn.Sequential(token_embedder, position_encoder)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)

        return out