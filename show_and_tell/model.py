import torch

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet152, ResNet152_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_dim, train_backbone=True):
        super().__init__()
        backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        modules = list(backbone.children())[:-1]  # CNN의 마지막 출력층은 제외
        self.backbone = nn.Sequential(*modules)

        if not train_backbone:
            # 백본 네트워크의 파라미터를 동결
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.linear = nn.Linear(backbone.fc.in_features, embed_dim)  # embedding layer
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)
        
    def forward(self, images):
        features = self.backbone(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))

        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers, dropout_prob=0.5, max_seq_length=20):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        embeddings = self.dropout(embeddings)  # Dropout on embeddings
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        outputs = self.dropout(outputs)  # Dropout on outputs
        
        return outputs
    
    def sample(self, features, states=None):
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted)
            
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        sampled_ids = torch.stack(sampled_ids, 1)

        return sampled_ids
