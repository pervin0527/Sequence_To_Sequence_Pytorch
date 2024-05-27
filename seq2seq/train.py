import os
import torch
import spacy
import evaluate
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtext.data.metrics import bleu_score
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.tensorboard import SummaryWriter

from model import Encoder, Decoder, Seq2Seq
from dataset import get_dataset, tokenize, numericalize, get_collate_fn, get_data_loader

def get_tokenizer_fn(tokenizer, lower):
    def tokenizer_fn(s):
        tokens = [token.text for token in tokenizer.tokenizer(s)]
        if lower:
            tokens = [token.lower() for token in tokens]
        return tokens

    return tokenizer_fn

def translate_sentence(
    sentence,
    model,
    en_tokenizer,
    de_tokenizer,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            tokens = [token.text for token in de_tokenizer.tokenizer(sentence)]
        else:
            tokens = [token for token in sentence]
        if lower:
            tokens = [token.lower() for token in tokens]
        tokens = [sos_token] + tokens + [eos_token]
        ids = de_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        context = model.encoder(tensor)
        hidden = context
        inputs = en_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden = model.decoder(inputs_tensor, hidden, context)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        tokens = en_vocab.lookup_tokens(inputs)
    return tokens

def init_weights(m):
    for name, param in m.named_parameters():
        torch.nn.init.normal_(param.data, mean=0, std=0.01)

def reverse_sequence(seq):
    return seq[::-1]

def train(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device, writer, epoch, reverse_input=False):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(data_loader, desc='Train', leave=False)):
        src = batch["de_ids"].to(device) ## src = [src length, batch size]
        trg = batch["en_ids"].to(device) ## trg = [trg length, batch size]

        if reverse_input:
            src = torch.flip(src, [0])
        
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio) ## output = [trg length, batch size, trg vocab size]
        
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim) ## output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1) ## trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        writer.add_scalar('Train/Loss', loss.item(), epoch * len(data_loader) + i)

    avg_loss = epoch_loss / len(data_loader)
    writer.add_scalar('Train/Perplexity', np.exp(avg_loss), epoch)
    
    return avg_loss

def eval(model, data_loader, criterion, device, writer, epoch, de_vocab, en_vocab, reverse_input=False):
    model.eval()
    epoch_loss = 0
    all_trg = []
    all_output = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc='Eval', leave=False)):
            src = batch["de_ids"].to(device) ## src = [src length, batch size]
            trg = batch["en_ids"].to(device) ## trg = [trg length, batch size]

            if reverse_input:
                src = torch.flip(src, [0])
            
            output = model(src, trg, 0) ## output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim) ## output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1) ## trg = [(trg length - 1) * batch size]
            
            loss = criterion(output, trg)
            epoch_loss += loss.item()

            # # Convert output and trg back to sequences for BLEU score calculation
            # output_seq = output.view(-1, trg.shape[0], output_dim).argmax(dim=-1).transpose(0, 1)
            # trg_seq = trg.view(-1, trg.shape[0]).transpose(0, 1)
            
            # for t, o in zip(trg_seq, output_seq):
            #     t_tokens = en_vocab.lookup_indices(t.cpu().numpy().tolist())
            #     o_tokens = en_vocab.lookup_indices(o.cpu().numpy().tolist())
            #     all_trg.append([t_tokens])
            #     all_output.append(o_tokens)

        avg_loss = epoch_loss / len(data_loader)
        # bleu = bleu_score(all_output, all_trg)
        writer.add_scalar('Eval/Loss', avg_loss, epoch)
        writer.add_scalar('Eval/Perplexity', np.exp(avg_loss), epoch)
        # writer.add_scalar('Eval/BLEU', bleu, epoch)

    # return avg_loss, bleu
    return avg_loss

def main():
    save_dir = "./runs"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)
    
    epochs = 20
    batch_size = 32
    clip = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder_embedding_dim = 256
    decoder_embedding_dim = 256
    hidden_dim = 512
    encoder_dropout = 0.5
    decoder_dropout = 0.5
    teacher_forcing_ratio = 0.5
    reverse_input = False

    lower = True
    min_freq = 2
    max_length = 20

    unk_token = "<unk>"
    pad_token = "<pad>"
    sos_token = "<sos>"
    eos_token = "<eos>"
    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token,
    ]

    en_tokenizer = spacy.load('en_core_web_sm')
    de_tokenizer = spacy.load('de_core_news_sm')
    train_dataset, valid_dataset, test_dataset = get_dataset()

    fn_kwargs = {
        "en_tokenizer": en_tokenizer,
        "de_tokenizer": de_tokenizer,
        "max_length": max_length,
        "lower": lower,
        "sos_token": sos_token,
        "eos_token": eos_token,
    }
    train_dataset = train_dataset.map(tokenize, fn_kwargs=fn_kwargs)
    valid_dataset = valid_dataset.map(tokenize, fn_kwargs=fn_kwargs)
    test_dataset = test_dataset.map(tokenize, fn_kwargs=fn_kwargs)

    en_vocab = build_vocab_from_iterator(train_dataset["en_tokens"], min_freq=min_freq, specials=special_tokens)
    de_vocab = build_vocab_from_iterator(train_dataset["de_tokens"], min_freq=min_freq, specials=special_tokens)
    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token]
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)
    input_dim = len(de_vocab)
    output_dim = len(en_vocab)

    fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}
    train_dataset = train_dataset.map(numericalize, fn_kwargs=fn_kwargs)
    valid_dataset = valid_dataset.map(numericalize, fn_kwargs=fn_kwargs)
    test_dataset = test_dataset.map(numericalize, fn_kwargs=fn_kwargs)

    data_type = "torch"
    format_columns = ["en_ids", "de_ids"]
    train_dataset = train_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)
    valid_dataset = valid_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)
    test_dataset = test_dataset.with_format(type=data_type, columns=format_columns, output_all_columns=True)

    train_dataloader = get_data_loader(train_dataset, batch_size=batch_size, pad_index=pad_index, shuffle=True)
    valid_dataloader = get_data_loader(valid_dataset, batch_size=batch_size, pad_index=pad_index, shuffle=False)
    test_dataloader = get_data_loader(test_dataset, batch_size=batch_size, pad_index=pad_index, shuffle=False)

    encoder = Encoder(input_dim, encoder_embedding_dim, hidden_dim, encoder_dropout)
    decoder = Decoder(output_dim, decoder_embedding_dim, hidden_dim, decoder_dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_index)

    best_valid_loss = float("inf")
    for epoch in range(1, epochs):
        print(f'\nEpoch : {epoch}')
        
        train_loss = train(model, train_dataloader, optimizer, criterion, clip, teacher_forcing_ratio, device, writer, epoch, reverse_input)
        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        
        # valid_loss, valid_bleu = eval(model, valid_dataloader, criterion, device, writer, epoch, de_vocab, en_vocab, reverse_input)
        valid_loss = eval(model, valid_dataloader, criterion, device, writer, epoch, de_vocab, en_vocab, reverse_input)
        # print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f} | Valid BLEU: {valid_bleu:7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{save_dir}/best.pt")

    writer.close()
    torch.save(model.state_dict(), f"{save_dir}/last.pt")

    sentence = test_dataset[0]["de"]
    expected_translation = test_dataset[0]["en"]
    translation = translate_sentence(
        sentence,
        model,
        en_tokenizer,
        de_tokenizer,
        en_vocab,
        de_vocab,
        lower,
        sos_token,
        eos_token,
        device,
    )

    print(sentence)
    print(expected_translation)
    print(translation)

    bleu = evaluate.load("bleu")
    translations = [
        translate_sentence(
            example["de"],
            model,
            en_tokenizer,
            de_tokenizer,
            en_vocab,
            de_vocab,
            lower,
            sos_token,
            eos_token,
            device,
        )
        for example in tqdm(test_dataset)
    ]
    predictions = [" ".join(translation[1:-1]) for translation in translations]
    references = [[example["en"]] for example in test_dataset]

    tokenizer_fn = get_tokenizer_fn(en_tokenizer, lower)
    results = bleu.compute(predictions=predictions, references=references, tokenizer=tokenizer_fn)
    print(results)


if __name__ == "__main__":
    main()
