# coding:utf-8
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

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
    '''
    input: n x 2048
    output: n x 3 x 224 x 224 0 ~ 1
    '''
    def __init__(self):
        super(Decoder, self).__init__()
        # 目标大小224x224，所以这里是14，然后依次放大14->28->56->112->224
        self.fc4 = nn.Linear(2048, 14 * 14 * 16)
        self.fc_bn4 = nn.BatchNorm1d(14 * 14 * 16)

        self.conv5 = nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

        self.relu = nn.ReLU() 

    def decode(self, z):
        z = self.fc4(z)
        fc4 = self.relu(self.fc_bn4(z).view(-1, 16, 14, 14))
        conv5 = self.relu(self.bn5(self.conv5(fc4)))
        conv6 = self.relu(self.bn6(self.conv6(conv5)))
        conv7 = self.relu(self.bn7(self.conv7(conv6)))
        r = self.conv8(conv7)
        # 此处用sigmoid限制输出到0~1之间(pytorch中的图像表示在0~1之间)
        r = F.sigmoid(r)
        return r

    def forward(self, z):
        return self.decode(z) 

class SimpleEncoder(nn.Module):
    '''
    input: n x 512 x 7 x 7 （VGG16特征）
    output: 
       mu: n x 2048
       logvar: n x 2048
    '''
    
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.vgg16_extractor = VGG16Extractor()
        self.fc1 = nn.Linear(512 * 7 * 7, 2048)
        self.fc2 = nn.Linear(512 * 7 * 7, 2048)
    
    def forward(self, x):
        x = self.vgg16_extractor(x)
        x = x.reshape(x.shape[0], -1)
        mu = self.fc1(x)
        logvar = self.fc2(x)
        return mu, logvar

class VAE_FashionEmbedding(nn.Module):

    def __init__(self, num_classes):
        super(VAE_FashionEmbedding, self).__init__()
        self.encoder = SimpleEncoder()
        self.decoder = Decoder()
        # self.ploss = VGG_perceptual_loss_16()
        self.finalfc = nn.Sequential(
            nn.Linear(2048, num_classes),
        )
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, img):
        '''
        img: n x 3 x 224 x 224
        '''
        mu, logvar = self.encoder(img)
        z = self.reparameterize(mu, logvar)
        img_decoded = self.decoder(z)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # TODO: embedding = mu 还是？
        embedding = mu

        # x = x.reshape(x.shape[0], -1)
        # mu = self.embedding(x)
        # sigma = self.embedding(x)
        # embedding = mu + exp(sigma / 2) * r
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        # img_decoded = self.decoder(embedding) 

        finalfc = self.finalfc(embedding)
        return {
            'embedding': embedding,
            'output': finalfc,
            'decoded': img_decoded,
            'KLD':KLD
        }