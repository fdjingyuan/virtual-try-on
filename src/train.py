from src.dataset import DeepFashionInShopDataset
from src.networks import VGGNet
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import os

if __name__ = '__main__':
	if os.path.exists('models') is False:
		os.makedirs('models')

	df = pd.read_csv(base_path + data/info.csv)
	
