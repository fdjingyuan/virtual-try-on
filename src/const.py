import time as time
import torch as torch
#from src.networks import FashionEmbedding as net
from src.vae import VAE_FashionEmbedding as net 

net_name = net.__name__
now_time = time.strftime('%m-%d %H:%M:%S', time.localtime())



USE_NET = net

NUM_CLASSES = 12923
FEATURE_EMBEDDING = 2048
STEP_SIZE = 10
WEIGHT_CENT = 1
LEARNING_RATE_DECAY = 0.5

# log 
TRAIN_DIR = 'runs/%s/' % net_name + now_time
VAL_DIR = 'runs/%s/' % net_name + now_time
MODEL_NAME = '%s.pkl' % net_name + now_time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


"""
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCH = 50

"""
