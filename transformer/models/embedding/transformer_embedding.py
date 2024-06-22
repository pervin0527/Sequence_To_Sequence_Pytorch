from torch import nn

class TransformerEmbedding(nn.Module):
    def __init__(self, word_embedding_layer, positional_encoding_layer, drop_prob=0):
        super().__init__()
        self.embedding = nn.Sequential(word_embedding_layer, positional_encoding_layer)
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        out = x
        out = self.embedding(out)
        out = self.dropout(out)

        return out