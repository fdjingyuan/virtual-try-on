#coding: utf-8
import torch
import torch.utils.data
import numpy as np
from torchvision import transforms
import pandas as pd
import cv2
import random
import skimage

class RandomFlip(object):

    def __call__(self, image):
        h, w = image.shape[:2]
        if np.random.rand() > 0.5:
            image = np.fliplr(image)

        return image


class CenterCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = int((h - new_h) / 2)
        left = int((w - new_w) / 2)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return image


class RandomRescale(object):

    def __init__(self, output_size_range):
        '''
        output_size_range指将短边缩放到的范围
        '''
        assert isinstance(output_size_range, tuple)
        self.lower_size = int(output_size_range[0])
        self.upper_size = int(output_size_range[1])

    def gen_output_size(self):
        return random.randint(self.lower_size, self.upper_size)

    def __call__(self, image):
        h, w = image.shape[:2]
        output_size = self.gen_output_size()
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        return img


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively

        return img



class DeepFashionInShopDataset(torch.utils.data.Dataset):

    def __init__(self, df, mode):
        '''
        mode:
            RANDOM：短边缩放到256->随机crop->随机flip
            CENTER：短边缩放到256->中心crop
        '''
        self.df = df
        
        self.rescale = Rescale(256)
        self.random_flip = RandomFlip()
        self.random_crop = RandomCrop((224, 224))
        self.center_crop = CenterCrop((224, 224))
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def plot_sample(self, i):
        import matplotlib.pyplot as plt
        sample = self[i]
        image = sample['raw_image']
        plt.figure(dpi=72)
        plt.imshow(image)

    def get_sample(self, i):
        sample = self[i]
        image = sample['image']
        return image


    def __getitem__(self, i):
        sample = self.df.iloc[i]
        image = cv2.imread(sample['image_name']) # h, w, c
        #image = io.imread(sample['image_name'])
        if self.mode == 'RANDOM':
            image = self.rescale(image)
            image = self.random_crop(image)
            image = self.random_flip(image)
        elif self.mode == 'CENTER':
            image = self.rescale(image)
            image = self.center_crop(image)
        else:
            raise NotImplementedError

        # support special numpy type
        # img_as_ubyte: 0-255
        #image = skimage.img_as_ubyte(image)
        # regenerate an image for ensure bug
        # BGR2RGB转换
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.copy()
        raw_image = image


        # convert to tensor and normalize
        # double -> float 
        image_tensor = self.to_tensor(raw_image) # between 0~1 
        # self.normalize is an inplace op so we use self.to_tensor to create a new tensor
        image = self.normalize(self.to_tensor(image))
        label = sample['item_id']

        ret = {
            'image': image, #tensor image for network to train
            'raw_image': raw_image, # raw image for plt to draw
            'label': label, #classification
            'image_tensor': image_tensor, # tensor image between 0~1
        }
        return ret
