import os
import cv2
import spacy
import torch
import pickle

from tqdm import tqdm
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
from torchvision import transforms
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator

nlp = spacy.load('en_core_web_sm')

def data_transform(is_train, img_size):
    if is_train:
        transform = transforms.Compose([transforms.RandomCrop((img_size, img_size)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    else:
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
        
    return transform
        

def yield_tokens(data_iter):
    for text in tqdm(data_iter, desc="Tokenizing captions"):
        yield [token.text for token in nlp(text)]

def build_vocab(caption_dir, save_path, min_freq=5):
    coco = COCO(caption_dir)
    counter = Counter()
    ids = coco.anns.keys()
    captions = [str(coco.anns[id]['caption']).lower() for id in ids]

    for tokens in yield_tokens(captions):
        counter.update(tokens)

    vocab = build_vocab_from_iterator([counter.elements()], specials=['<unk>', '<pad>', '<sos>', '<eos>'], min_freq=min_freq)
    vocab.set_default_index(vocab['<unk>'])

    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(vocab, f)

    return vocab

def resize_images(image_dir, image_size, save_dir):
    images = os.listdir(image_dir)
    os.makedirs(save_dir, exist_ok=True)
    for file_name in tqdm(images, desc="Resizing images"):
        image = cv2.imread(f"{image_dir}/{file_name}")
        image = cv2.resize(image, (image_size, image_size))
        cv2.imwrite(f'{save_dir}/{file_name}', image)

class CocoDataset(Dataset):
    def __init__(self, root, json, vocab, transform=None):
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab

        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(f"{self.root}/{path}")
        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        tokens = [token.text for token in nlp(str(caption).lower())]
        caption = []
        caption.append(vocab.lookup_indices(['<sos>'])[0])
        caption.extend([vocab.lookup_indices([token])[0] for token in tokens])
        caption.append(vocab.lookup_indices(['<eos>'])[0])
        target = torch.Tensor(caption)

        return image, target

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths

if __name__ == "__main__":
    caption_path = '/home/pervinco/Datasets/COCO2017/annotations/captions_train2017.json'
    image_path = '/home/pervinco/Datasets/COCO2017/train2017'
    min_freq = 5
    image_size = 256
    
    img_save_path = '/home/pervinco/Datasets/COCO2017/train_images'
    vocab_save_path = '/home/pervinco/Datasets/COCO2017/annotations/vocab.pkl'

    # vocab = build_vocab(caption_path, vocab_save_path, min_freq)
    # resize_images(image_path, image_size, img_save_path)
    resize_images(image_path.replace('train', 'val'), image_size, img_save_path.replace('train', 'val'))

    with open(vocab_save_path, 'rb') as f:
        vocab = pickle.load(f)

    dataset = CocoDataset(image_path, json=caption_path, vocab=vocab)
    image, target = dataset[0]
    print(image.shape)
    print(target.shape)
    cv2.imwrite('./test/sample.jpg', image)