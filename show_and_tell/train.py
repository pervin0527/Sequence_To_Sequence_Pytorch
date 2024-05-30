import os
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter

from model import EncoderCNN, DecoderRNN
from dataset import CocoDataset, collate_fn, build_vocab, resize_images, data_transform

def train(encoder, decoder, dataloader, optimizer, criterion, device):
    encoder.train()
    decoder.train()

    total_loss = 0
    total_perplexity = 0
    total_count = 0

    for images, captions, seq_len in tqdm(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, seq_len, batch_first=True)[0]

        features = encoder(images)
        outputs = decoder(features, captions, seq_len)
        loss = criterion(outputs, targets)

        decoder.zero_grad()
        encoder.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_count += targets.size(0)
        perplexity = torch.exp(loss).item()
        total_perplexity += perplexity

    avg_loss = total_loss / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    
    return avg_loss, avg_perplexity

def valid(encoder, decoder, dataloader, criterion, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_perplexity = 0
    total_count = 0

    with torch.no_grad():
        for images, captions, seq_len in tqdm(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, seq_len, batch_first=True)[0]

            features = encoder(images)
            outputs = decoder(features, captions, seq_len)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_count += targets.size(0)
            perplexity = torch.exp(loss).item()
            total_perplexity += perplexity

    avg_loss = total_loss / len(dataloader)
    avg_perplexity = total_perplexity / len(dataloader)
    
    return avg_loss, avg_perplexity

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "./runs"
    os.makedirs(save_dir, exist_ok=True)

    data_dir = "/home/pervinco/Datasets/COCO2017"
    train_vocab_dir = f'{data_dir}/ImageCaption/vocab.pkl'
    train_img_dir = f"{data_dir}/ImageCaption/train_images"
    train_cap_dir = f"{data_dir}/annotations/captions_train2017.json"

    epochs = 100
    img_size = 224
    batch_size = 1024
    learning_rate = 0.001
    workers = 4

    min_freq = 3
    embed_dim = 512
    hidden_dim = 512
    num_layers = 1
    max_length = 100

    if not os.path.exists(train_vocab_dir) and not os.path.exists(train_img_dir):
        vocab = build_vocab(caption_dir=f'{data_dir}/annotations/captions_train2017.json',
                            save_path=f'{data_dir}/ImageCaption/vocab.pkl',
                            min_freq=min_freq)
        resize_images(image_dir=f"{data_dir}/train2017",
                      image_size=img_size,
                      save_dir=f'{data_dir}/ImageCaption/images')
    else:
        with open(f"{data_dir}/ImageCaption/vocab.pkl", 'rb') as f:
            vocab = pickle.load(f)

    train_transform = data_transform(True, img_size)
    train_dataset = CocoDataset(train_img_dir, train_cap_dir, vocab, train_transform)
    valid_dataset = CocoDataset(train_img_dir.replace('train', 'val'), train_cap_dir.replace('train', 'val'), vocab, train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, collate_fn=collate_fn)

    encoder = EncoderCNN(embed_dim).to(device)
    decoder = DecoderRNN(embed_dim, hidden_dim, len(vocab), num_layers, max_length).to(device)
    criterion = nn.CrossEntropyLoss()
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    writer = SummaryWriter(log_dir=save_dir)
    best_valid_perplexity = float('inf')

    for epoch in range(1, epochs+1):
        print(f"\nEpoch : [{epoch}/{epochs}]")

        train_loss, train_perplexity = train(encoder, decoder, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_perplexity = valid(encoder, decoder, valid_dataloader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}, Valid Perplexity: {valid_perplexity:.4f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Perplexity/train', train_perplexity, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Perplexity/valid', valid_perplexity, epoch)

        if valid_perplexity < best_valid_perplexity:
            best_valid_perplexity = valid_perplexity
            torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder-best.pth'))
            torch.save(decoder.state_dict(), os.path.join(save_dir, 'decoder-best.pth'))

    torch.save(encoder.state_dict(), os.path.join(save_dir, 'encoder-last.pth'))
    torch.save(decoder.state_dict(), os.path.join(save_dir, 'decoder-last.pth'))
    writer.close()

if __name__ == "__main__":
    main()
