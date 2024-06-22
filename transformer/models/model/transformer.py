import torch
import numpy as np

from torch import nn
from torch.nn import functional as F

class Transformer(nn.Module):
    def __init__(self, src_embed, trg_embed, encoder, decoder, generator):
        super().__init__()
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator


    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)


    def decode(self, trg, encoder_out, trg_mask, src_trg_mask):
        return self.decoder(self.trg_embed(trg), encoder_out, trg_mask, src_trg_mask)


    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        src_trg_mask = self.make_src_trg_mask(src, trg)
        encoder_out = self.encode(src, src_mask)
        decoder_out = self.decode(trg, encoder_out, trg_mask, src_trg_mask)
        out = self.generator(decoder_out)
        out = F.log_softmax(out, dim=-1)

        return out, decoder_out


    def make_src_mask(self, src):
        pad_mask = self.make_pad_mask(src, src)
        
        return pad_mask


    def make_trg_mask(self, trg):
        pad_mask = self.make_pad_mask(trg, trg)
        seq_mask = self.make_subsequent_mask(trg, trg)

        return pad_mask & seq_mask


    def make_src_trg_mask(self, src, trg):
        pad_mask = self.make_pad_mask(trg, src)

        return pad_mask


    def make_pad_mask(self, query, key, pad_idx=1):
        """
        Padding Mask
            query: (n_batch, query_seq_len)
            key: (n_batch, key_seq_len)
        """

        query_seq_len, key_seq_len = query.size(1), key.size(1) ## query_seq_len, key_seq_len

        ## ne : pad_idx가 아닌 원소들을 찾아 True/False인 텐서를 만든다.
        key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)

        ## 세번째 차원을 쿼리 시퀀스 길이에 맞춰 반복해서 쌓는다.
        key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

        ## ne : pad_idx가 아닌 원소들을 찾아 True/False인 텐서를 만든다.
        query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
        query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

        mask = key_mask & query_mask ## 두 행렬에서 True인 원소만 True로.
        mask.requires_grad = False

        return mask


    def make_subsequent_mask(self, query, key):
        """
        Look-Ahead Mask
            query : (batch_size, query_seq_len)
            key : (batch_size, key_seq_len)
        """
        query_seq_len, key_seq_len = query.size(1), key.size(1)

        ## shape이 query_seq_len, key_seq_len인 lower-triangular matrix를 만든다. k는 주대각원소의 위쪽에 있는 원소값.
        tril = np.tril(np.ones((query_seq_len, key_seq_len)), k=0).astype('uint8') # lower triangle without diagonal
        mask = torch.tensor(tril, dtype=torch.bool, requires_grad=False, device=query.device) ## boolean type의 텐서로 변환.

        return mask
