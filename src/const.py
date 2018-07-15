import time as _time
import torch as _torch
from src.networks import FashionEmbedding as _net

_name = 'vgg'
_time = _time.strftime('%m-%d %H:%M:%S', _time.localtime())

NUM_CLASSES = 12923

USE_NET = _net
# log 
TRAIN_DIR = 'runs/%s/' % _name + _time
VAL_DIR = 'runs/%s/' % _name + _time

MODEL_NAME = '%s.pkl' % _name

device = _torch.device('cuda:0' if _torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCH = 50
LEARNING_RATE_DECAY = 0.98


WEIGHT_CENT = 1