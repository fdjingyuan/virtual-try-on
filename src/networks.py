import torch
import torch.nn as nn
from torch.nn import init
import torchvision


class ModuleWithAttr(nn.Module):

    # 只能是数字，默认注册为0

    def __init__(self, extra_info=['step']):
        super(ModuleWithAttr, self).__init__()
        for key in extra_info:
            self.set_buffer(key, 0)

    def set_buffer(self, key, value):
        if not(hasattr(self, '__' + key)):
            self.register_buffer('__' + key, torch.tensor(value))
        setattr(self, '__' + key, torch.tensor(value))

    def get_buffer(self, key):
        if not(hasattr(self, '__' + key)):
            raise Exception('no such key!')
        return getattr(self, '__' + key).item()



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

class FashionEmbedding(ModuleWithAttr):
    
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
