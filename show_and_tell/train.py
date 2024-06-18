import os
import json
import torch

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence

from model import Encoder, Decoder
from dataset import CaptionDataset

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(scores, targets, k):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder': encoder,
        'decoder': decoder,
        'encoder_optimizer': encoder_optimizer,
        'decoder_optimizer': decoder_optimizer
    }
    directory = os.path.join('./runs', data_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, 'checkpoint_' + data_name + '.pth.tar')
    torch.save(state, filename)

    if is_best:
        torch.save(state, os.path.join(directory, 'BEST_' + filename))


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, grad_clip, device):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
        targets = caps_sorted[:, 1:]

        scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        scores_packed_data = scores_packed.data.to(device)
        targets_packed_data = targets_packed.data.to(device)

        loss = criterion(scores_packed_data, targets_packed_data)

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        top5 = accuracy(scores_packed_data, targets_packed_data, 5)
        losses.update(loss.item(), targets_packed_data.size(0))
        top5accs.update(top5, targets_packed_data.size(0))

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time, loss=losses,
                                                                          top5=top5accs))


def validate(val_loader, encoder, decoder, criterion, word_map, device):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    references = list()
    hypotheses = list()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)

            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, sort_ind = decoder(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            scores_packed_data = scores_packed.data.to(device)
            targets_packed_data = targets_packed.data.to(device)

            loss = criterion(scores_packed_data, targets_packed_data)

            losses.update(loss.item(), targets_packed_data.size(0))
            top5 = accuracy(scores_packed_data, targets_packed_data, 5)
            top5accs.update(top5, targets_packed_data.size(0))

            if i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time, loss=losses, top5=top5accs))

            sort_ind = sort_ind.to('cpu')
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(map(lambda c: [w for w in c if w not in {word_map['<sos>'], word_map['<pad>']}], img_caps))
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu1 = corpus_bleu(references, hypotheses, weights=[1, 0, 0, 0]) * 100
        bleu2 = corpus_bleu(references, hypotheses, weights=[1/2, 1/2, 0, 0]) * 100
        bleu3 = corpus_bleu(references, hypotheses, weights=[1/3, 1/3, 1/3, 0]) * 100
        bleu4 = corpus_bleu(references, hypotheses, weights=[1/4, 1/4, 1/4, 1/4]) * 100

        print('\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}\n'.format(loss=losses, top5=top5accs))
        print(' * BLEU-1 - {bleu1}, BLEU-2 - {bleu2}, BLEU-3 - {bleu3}, BLEU-4 - {bleu4}\n'.format(bleu1=bleu1, bleu2=bleu2, bleu3=bleu3, bleu4=bleu4))

    return bleu4


def main():
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    decoder = Decoder(embed_dim=emb_dim, decoder_dim=decoder_dim, vocab_size=len(word_map), dropout=dropout)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)
    
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None

    decoder = decoder.to(device)
    encoder = encoder.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_dataset = CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([normalize]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    valid_dataset = CaptionDataset(data_folder, data_name, 'VAL', transform=transforms.Compose([normalize]))
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    best_bleu4 = 0.
    for epoch in range(1, epochs+1):
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch,
              grad_clip=grad_clip,
              device=device)

        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                word_map=word_map,
                                device=device)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer, recent_bleu4, is_best)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_folder = '/home/pervinco/Datasets/ImageCaption_Dataset'
    data_name = 'flickr8k_5_cap_per_img_5_min_word_freq'

    emb_dim = 512  # dimension of word embeddings
    attention_dim = 512  # dimension of attention linear layers
    decoder_dim = 512  # dimension of decoder RNN
    dropout = 0.5

    epochs = 10
    batch_size = 32
    encoder_lr = 1e-4
    decoder_lr = 4e-4
    grad_clip = 5.
    alpha_c = 1.
    best_bleu4 = 0.
    fine_tune_encoder = False

    main()