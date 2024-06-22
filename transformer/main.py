import os
import torch
import logging

from tqdm import tqdm
from torch import nn, optim
from torchtext.data.metrics import bleu_score

from config import *
from data import Multi30kDataset
from models.build_model import build_model


def get_bleu_score(output, gt, vocab, specials, max_n=4):
    def itos(x):
        x = list(x.cpu().numpy())
        tokens = vocab.lookup_tokens(x)
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in list(specials.keys()), tokens))
        return tokens

    pred = [out.max(dim=1)[1] for out in output]
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))

    return  bleu_score(pred_str, gt_str, max_n=max_n) * 100.0


def greedy_decode(model, src, max_len, start_symbol, end_symbol):
    src = src.to(model.device)
    src_mask = model.make_src_mask(src).to(model.device)
    memory = model.encode(src, src_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(model.device)
    for i in range(max_len-1):
        memory = memory.to(model.device)
        trg_mask = model.make_trg_mask(ys).to(model.device)
        src_trg_mask = model.make_src_trg_mask(src, ys).to(model.device)
        out = model.decode(ys, memory, trg_mask, src_trg_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == end_symbol:
            break
        
    return ys


def train(model, data_loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for (src, trg) in tqdm(data_loader, desc="train", leave=False):
        src = src.to(DEVICE)
        trg = trg.to(DEVICE)
        trg_x = trg[:, :-1]
        trg_y = trg[:, 1:]

        optimizer.zero_grad()

        output, _ = model(src, trg_x)

        y_hat = output.contiguous().view(-1, output.shape[-1])
        y_gt = trg_y.contiguous().view(-1)
        loss = criterion(y_hat, y_gt)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(list(data_loader))


def evaluate(model, data_loader, criterion, trg_vocab, specials):
    model.eval()
    epoch_loss = 0

    total_bleu = []
    with torch.no_grad():
        for (src, trg) in tqdm(data_loader, desc="eval", leave=False):
            src = src.to(DEVICE)
            trg = trg.to(DEVICE)
            trg_x = trg[:, :-1]
            trg_y = trg[:, 1:]

            output, _ = model(src, trg_x)

            y_hat = output.contiguous().view(-1, output.shape[-1])
            y_gt = trg_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, trg_y, trg_vocab, specials)
            total_bleu.append(score)

    loss_avr = epoch_loss / len(list(data_loader))
    bleu_score = sum(total_bleu) / len(total_bleu)

    return loss_avr, bleu_score


def initialize_weights(model):
    if hasattr(model, 'weight') and model.weight.dim() > 1:
        nn.init.kaiming_uniform_(model.weight.data)


def main():
    dataset = Multi30kDataset(data_dir=f"{DATA_DIR}/Multi30k", source_language=SRC_LANGUAGE,  target_language=TGT_LANGUAGE,  max_seq_len=MAX_SEQ_LEN, vocab_min_freq=2)
    train_iter, valid_iter, test_iter = dataset.get_iter(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(len(dataset.src_vocab), len(dataset.trg_vocab), device=DEVICE, drop_prob=DROP_PROB)
    model.apply(initialize_weights)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True, factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.PAD_IDX)

    print("\nTrain Start")
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)

    min_val_loss = 0
    for epoch in range(EPOCHS):
        train_loss = train(model, train_iter, optimizer, criterion)
        valid_loss, bleu_scores  = evaluate(model, valid_iter, criterion, dataset.trg_vocab, dataset.SPECIALS)

        if epoch == 0:
            min_val_loss = valid_loss

        if epoch > 1:
            if valid_loss < min_val_loss:
                min_val_loss = valid_loss
                ckpt = f"{SAVE_DIR}/{epoch:04}.pt"
                torch.save({'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss,
                            'val_loss' : valid_loss}, ckpt)

        if epoch > WARM_UP_STEP:
            scheduler.step(valid_loss)

        print(f"Epoch : {epoch + 1} | train_loss: {train_loss:.5f} valid_loss: {valid_loss:.5f}, bleu_scores: {bleu_scores:.5f}")
        print("Predict : ", dataset.translate(model, "A little girl climbing into a wooden playhouse .", greedy_decode))
        print(f"Answer : Ein kleines Mädchen klettert in ein Spielhaus aus Holz . \n")

    test_loss, bleu_scores = evaluate(model, test_iter, criterion)
    print(f"test_loss: {test_loss:.5f}")
    print(f"bleu_scores: {bleu_scores:.5f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    logging.basicConfig(level=logging.INFO)
    main()
