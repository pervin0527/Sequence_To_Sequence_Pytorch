import math
import torch

from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, max_seq_len=256, device=torch.device("cpu")):
        super().__init__()
        encoded = torch.zeros(max_seq_len, d_embed)
        encoded.requires_grad = False

        position = torch.arange(0, max_seq_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) * -(math.log(10000.0) / d_embed))

        encoded[:, 0::2] = torch.sin(position * div_term)
        encoded[:, 1::2] = torch.cos(position * div_term)
        self.encoded = encoded.unsqueeze(0).to(device)

    def forward(self, x):
        _, seq_len, _ = x.size()
        pos_embed = self.encoded[:, :seq_len, :]
        out = x + pos_embed

        return out