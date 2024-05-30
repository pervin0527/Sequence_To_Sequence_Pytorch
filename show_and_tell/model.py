import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet152, ResNet152_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1] ## CNN의 마지막 출력층은 제외
        self.backbone = nn.Sequential(*modules)

        self.linear = nn.Linear(backbone.fc.in_features, embed_dim) ## embedding layer
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)

        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features

    
class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers, max_seq_length=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size) ## output layer
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions) ## embedding된 토큰 문장.

        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) ## encoder가 임베딩한 벡터와 임베딩된 caption을 cat
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) ## 패딩을 적용해서 차원을 맞춰줌.
        hiddens, _ = self.lstm(packed) ## 다음 hidden state 계산.
        outputs = self.linear(hiddens[0])

        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states) ## hiddens : (batch_size, 1, hidden_dim)
            outputs = self.linear(hiddens.squeeze(1)) ## outputs :  (batch_size, vocab_size)
            _, predicted = outputs.max(1) ## predicted: (batch_size) 확률이 가장 높은 하나를 선정.
            sampled_ids.append(predicted)

            inputs = self.embed(predicted) ## inputs: (batch_size, embed_dim)
            inputs = inputs.unsqueeze(1) ## inputs: (batch_size, 1, embed_dim)
        sampled_ids = torch.stack(sampled_ids, 1) ## sampled_ids: (batch_size, max_seq_length)

        return sampled_ids