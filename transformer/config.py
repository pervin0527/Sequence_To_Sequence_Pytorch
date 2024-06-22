import os
import torch

DATA_DIR = "/home/pervinco/Datasets"
SAVE_DIR = "/home/pervinco/Models/Transformers"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

EPOCHS = 100
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-9
ADAM_EPS = 5e-9
SCHEDULER_FACTOR = 0.9
SCHEDULER_PATIENCE = 10
WARM_UP_STEP = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min([os.cpu_count(), BATCH_SIZE if BATCH_SIZE > 1 else 0, 8])

D_MODEL = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FFN_DIM = 2048
MAX_SEQ_LEN = 256
DROP_PROB = 0.1