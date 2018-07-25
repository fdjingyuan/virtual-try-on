import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.utils.model_zoo as model_zoo
from 


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

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc4 = nn.Linear(2048, 8 * 8 * 16)
        self.fc_bn4 = nn.BatchNorm1d(8 * 8 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU() 

    def decode(self, z):
        fc4 = self.relu(self.fc_bn4(z).view(-1, 16, 8, 8)

        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        return self.conv8(conv7).view(-1, 3, 32, 32)

    def forward(self, z):
        return self.decode(z) 



class VAE_FashionEmbedding(nn.Module):
    
    def __init__(self, num_classes):
        super(FashionEmbedding, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.decoder = Decoder()
        self.ploss = VGG_perceptual_loss_16()
        self.embedding = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), 
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 2048),
        )
        self.finalfc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )

    def forward(self, img, r):
        x = self.vgg16_extractor(img)
        # (batchsize, 512,7,7) -> (batchsize, 512*7*7)
        x = x.reshape(x.shape[0], -1)
        mu = self.embedding(x)
        sigma = self.embedding(x)

        embedding = mu + exp(sigma / 2) * r
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        img_decoded = self.decoder(embedding) 
       

        finalfc = self.finalfc(embedding)
        return {
            'embedding': embedding,
            'output': finalfc,
            'decoded': img_decoded,
            'KLD':KLD

        }



