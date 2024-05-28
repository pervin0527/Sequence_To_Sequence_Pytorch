import os
import cv2
import spacy
import pickle

from tqdm import tqdm
from collections import Counter
from pycocotools.coco import COCO
from torchtext.vocab import build_vocab_from_iterator

nlp = spacy.load('en_core_web_sm')

def yield_tokens(data_iter):
    for text in data_iter:
        yield [token.text for token in nlp(text)]


def build_vocab(caption_dir, min_freq=5):
    coco = COCO(caption_dir)
    counter = Counter()
    ids = coco.anns.keys()
    captions = [str(coco.anns[id]['caption']).lower() for id in ids]

    for tokens in yield_tokens(captions):
        counter.update(tokens)

    vocab = build_vocab_from_iterator(counter.items(), specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=min_freq)
    vocab.set_default_index(vocab['<unk>'])

    save_dir = "/".join(caption_dir.split('/')[:-1]) + '/vocab.pkl'
    save_dir = "./test/vocab.pkl"
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir, 'wb') as f:
        pickle.dump(save_dir, f)

    return vocab


def resize_images(image_dir, image_size):
    images = os.listdir(image_dir)
    num_images = len(images)

    save_dir = "/".join(image_dir.split('/')[:-1])
    save_dir = "./test"
    os.makedirs(save_dir, exist_ok=True)
    for image in tqdm(images):
        print(image)
        image = cv2.imread(image)
        image = cv2.resize(image, (image_size, image_size))
        # cv2.imwrite()


if __name__ == "__main__":
    caption_path = '/home/pervinco/Datasets/COCO2017/annotations/captions_train2017.json'
    image_path = '/home/pervinco/Datasets/COCO2017/train2017'
    min_freq = 5
    image_size = 256

    vocab = build_vocab(caption_path, min_freq)
    resize_images(image_path, image_size)
