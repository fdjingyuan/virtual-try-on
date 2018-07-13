import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataset import DeepFashionInShopDataset
import torchvision

class VGGNet(nn.Module):
	def __init__(self, features, num_classes = 1000, init_weights = True):
		super(VGGNet,self).__init__()
		self.features = features
		self.classifier = nn.Sequential(
			nn.Linear(512 * 7 * 7, 4096),
			nn.ReLu(True),
			nn.Dropout(),
			nn.Linear(4096,4096),
			nn.ReLU(True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
			)
		if init_weights:
			self._initialize_weights()

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)

