import time as time
import torch as torch
from src.networks import FashionEmbedding as net

net_name = net.__name__
now_time = time.strftime('%m-%d %H:%M:%S', time.localtime())



USE_NET = net

NUM_CLASSES = 12923

# log 
TRAIN_DIR = 'runs/%s/' % net_name + now_time
VAL_DIR = 'runs/%s/' % net_name + now_time

MODEL_NAME = '%s.pkl' % net_name

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCH = 50
LEARNING_RATE_DECAY = 0.98
"""
