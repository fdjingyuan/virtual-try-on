import torch
import torch.nn as nn
from torch.nn import init
import torchvision


class VGG16Extractor(nn.Module):
    
    def __init__(self):
        super(VGG16Extractor, self).__init__()
        # features: conv layers results
        self.vgg = torchvision.models.vgg16(pretrained=True).features

    def forward(self, x):
        # a series of Conv2d ReLU MaxPool2d
        for name, layer in self.vgg._modules.items():
            x = layer(x)
        return x

class FashionEmbedding(nn.Module):
    
    def __init__(self, num_classes):
        super(FashionEmbedding, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.embedding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
        )
        self.softmax = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
    
    def forward(self, x):
        x = self.vgg16_extractor(x)
        # (batchsize, 512,7,7) -> (batchsize, 512*7*7)
        x = x.reshape(x.shape[0], -1)
        embedding = self.embedding(x)
        softmax = self.softmax(embedding)
        return {
            'embedding': embedding,
            'output': softmax,
        }
