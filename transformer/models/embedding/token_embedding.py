import math
import torch.nn as nn

## Word Embedding
class WordEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) ## 임베딩을 학습 과정에서 조정하는 임베딩 층.
        self.d_model = d_model

    def forward(self, x):
        ## 임베딩 벡터에서 값이 너무 큰 원소를 억제하기 위해 math.sqrt(self.d_model)를 곱한다.
        out = self.embedding(x) * math.sqrt(self.d_model)

        return out