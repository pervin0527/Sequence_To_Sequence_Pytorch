from torch import nn

class FeedForwardLayer(nn.Module):
    def __init__(self, d_embed, d_ff, drop_prob=0):
        super().__init__()
        self.fc1 = nn.Linear(d_embed, d_ff)   # (d_embed, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.fc2 = nn.Linear(d_ff, d_embed) # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out