import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
from abc import ABC, abstractmethod
from PIL import Image

class BaseDataset(data.Dataset):
    def __init__(self, conf):
        self.conf = conf
        self.root = conf.dataroot
    
    @abstractmethod
    def __len__(self):
        return 0
    
    @abstractmethod
    def __getitem__(self, idx):
        pass

def get_transform(conf, is2d=True, grayscale=False, method=Image.BICUBIC):
    transforms_list = []
    if grayscale:
        transforms_list.append(transforms.Grayscale(1))
    if 'resize' in conf.preprocess:
        osize = [conf.load_size, conf.load_size]
        transforms_list.append(transforms.Resize(osize, method))
    if 'centercrop' in conf.preprocess:
        transforms_list.append(transforms.CenterCrop(conf.crop_size))
    if 'crop' in conf.preprocess:
        transforms_list.append(transforms.RandomCrop(conf.crop_size))


    if conf.preprocess == 'none':
        transforms_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
    
    if conf.flip:
        transforms_list.append(transforms.RandomHorizontalFlip())
    
    transforms_list.append(transforms.ToTensor())
    if is2d:
        transforms_list.append(transforms.Normalize((0.5, ), (0.5,)))
    else:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return transforms.Compose(transforms_list)

def transform_resize(conf, method=Image.BICUBIC):
    transforms_list = []
    osize = [conf.load_size, conf.load_size]
    transforms_list.append(transforms.Resize(osize, method))
    return transforms.Compose(transforms_list)



def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh/base)*base)
    w = int(round(ow/base)*base)
    if h == oh and w == ow:
        return img
    return img.resize((w, h), method)


def normalize(image):
    return (image-image.mean())/image.std()

def standard(image): # range in [-1, 1]
    return (image - image.mean())/(image.max() - image.min())

