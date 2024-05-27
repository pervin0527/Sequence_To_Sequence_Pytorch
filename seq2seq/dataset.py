import torch
import spacy
import datasets

from tqdm import tqdm
from collections import Counter

from torchtext.vocab import vocab
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def get_dataset():
    multi30k = datasets.load_dataset("bentrevett/multi30k")
    train_dataset, valid_dataset, test_dataset = multi30k['train'], multi30k['validation'], multi30k['test']

    return train_dataset, valid_dataset, test_dataset


def tokenize(example, en_tokenizer, de_tokenizer, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_tokenizer.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in de_tokenizer.tokenizer(example["de"])][:max_length]

    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]

    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {"en_tokens": en_tokens, "de_tokens": de_tokens}


def numericalize(example, en_vocab, de_vocab):
    en_ids = en_vocab.lookup_indices(example["en_tokens"])
    de_ids = de_vocab.lookup_indices(example["de_tokens"])
    return {"en_ids": en_ids, "de_ids": de_ids}


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_en_ids = [example["en_ids"] for example in batch]
        batch_de_ids = [example["de_ids"] for example in batch]
        batch_en_ids = torch.nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
        batch_de_ids = torch.nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
        batch = {
            "en_ids": batch_en_ids,
            "de_ids": batch_de_ids,
        }
        return batch

    return collate_fn


def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )

    return data_loader