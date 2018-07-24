import torch
import torch.nn as nn
from torch.nn import init
import torchvision

def make_layers(cfg, batch_norm=True, in_channels = 3, norm = nn.BatchNorm2d):
    layers = []
    in_channels = in_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, str):
            if v[0] == 'D':
                output_channels = int(v[1:])
                conv = nn.ConvTranspose2d(in_channels, output_channels,
                                            kernel_size=4, stride=2,padding=1)
                if batch_norm:
                    layers += [conv, norm(output_channels), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = output_channels
            elif v[0] == 'T':
                output_channels = int(v[1:])
                conv = nn.Conv2d(in_channels, output_channels,
                                            kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv, norm(output_channels), nn.Tanh()]
                else:
                    layers += [conv, nn.Tanh()]
                in_channels = output_channels
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_fc_layers(cfg, batch_norm=False, in_channels = 26):
    layers = []
    in_channels = in_channels
    for v in cfg:
        layers += [nn.Linear(in_channels, v), nn.BatchNorm1d(v), nn.ReLU()]
        in_channels = v
    return nn.Sequential(*layers)

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

class VGG_perceptual_loss_19_fashion(nn.Module):
    def __init__(self):
        super(VGG_perceptual_loss_19_fashion, self).__init__()
        cfg1 = [64, 64]
        cfg2 = ['M',128,128]
        cfg3 = ['M',256,256]
        cfg4 = [256,256,'M',512,512]
        cfg5 = [512,512,'M',512,512]
        self.net1 = make_layers(cfg1, False, 3)
        self.net2 = make_layers(cfg2, False, 64)
        self.net3 = make_layers(cfg3, False, 128)
        self.net4 = make_layers(cfg4, False, 256)
        self.net5 = make_layers(cfg5, False, 512)
        model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        state_dict = model_zoo.load_url(model_url)
        keys = state_dict.keys()
        self.init_weight(self.net1, state_dict, keys[0*2:2*2])
        self.init_weight(self, keys[2*2:4*2])
        self.init_weight(self.net3, state_dict, keys[4*2:6*2])
        self.init_weight(self.net4, state_dict, keys[6*2:10*2])
        self.init_weight(self.net5, state_dict, keys[10*2:14*2])
        self.mean = Variable(torch.from_numpy(np.array([0.485, 0.456, 0.406])).reshape(1,3,1,1)).cuda(0).float()
        self.var = Variable(torch.from_numpy(np.array([0.229, 0.224, 0.225])).reshape(1,3,1,1)).cuda(0).float()
        
    def init_weight(self, net, state_dict, keys):
        length = len(keys)
        i = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = state_dict[keys[i]]
                m.weight.bias = state_dict[keys[i + 1]]
                i += 2

    def get_feat(self, img):
        if self.mean.get_device() != img.get_device():
            self.mean = self.mean.cuda(img.get_device())
            self.var = self.var.cuda(img.get_device())
        img = ((img * 0.5 + 0.5) - self.mean) / self.var
        feat1 = self.net1(img)
        feat2 = self.net2(feat1)
        feat3 = self.net3(feat2)
#        feat4 = self.net4(feat3)
#        feat5 = self.net5(feat4)
        return [img, feat1, feat2, feat3]

    def forward(self, img1, img2):
        feat1_list = self.get_feat(img1)
        feat2_list = self.get_feat(img2)
        loss = 0
        for i in range(len(feat1_list)):
            loss += torch.abs(feat1_list[i] - feat2_list[i]).mean()
#        loss += 20 * torch.abs(img1 - img2).mean()
        return loss

class VGG_perceptual_loss_16(nn.Module):
    def __init__(self):
        super(VGG_perceptual_loss_16, self).__init__()
        cfg1 = [64, 64]
        cfg2 = ['M',128,128]
        cfg3 = ['M',256,256,256]
        self.net1 = make_layers(cfg1, False, 3)
        self.net2 = make_layers(cfg2, False, 64)
        self.net3 = make_layers(cfg3, False, 128)
        model_url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        state_dict = model_zoo.load_url(model_url)
        keys = state_dict.keys()
        self.init_weight(self.net1, state_dict, keys[0:4])
        self.init_weight(self.net2, state_dict, keys[4:8])
        self.init_weight(self.net3, state_dict, keys[8:14])

    def init_weight(self, net, state_dict, keys):
        length = len(keys)
        i = 0
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = state_dict[keys[i]]
                m.weight.bias = state_dict[keys[i + 1]]
                i += 2

    def forward(self, img1, img2):
        feat1_1 = self.net1(img1)
        feat2_1 = self.net2(feat1_1)
        feat3_1 = self.net3(feat2_1)
        feat1_2 = self.net1(img2)
        feat2_2 = self.net2(feat1_2)
        feat3_2 = self.net3(feat2_2)
        return torch.abs(feat1_1 - feat1_2).mean() + torch.abs(feat2_1 - feat2_2).mean() +  torch.abs(feat3_1 - feat3_2).mean()

class FashionEmbedding(nn.Module):
    
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
        embedding = mu + sigma * r
        img_decoded = self.decoder(embedding) 
        ploss = self.ploss(img, img_decoded)

        finalfc = self.finalfc(embedding)
        return {
            'embedding': embedding,
            'output': finalfc,
            'decoded': img_decoded,
            'ploss': ploss
        }



