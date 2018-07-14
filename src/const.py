import time
import torch
name = time.strftime('%m-%d %H:%M:%S', time.localtime())


base_path = '/home/jyliu/try-on-codes'
#TRAIN_DIR = 'runs/' + name
#VAL_DIR = 'runs/' + name


USE_NET = 'vgg16'
MODEL_NAME = USE_NET+'.pkl'

NUM_EPOCH = 20
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9
BATCH_SIZE = 16
