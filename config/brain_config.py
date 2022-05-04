import torch as t
import warnings
from config.base_config import BaseConfig

class BrainConfig(BaseConfig):
    # preprocess
    load_size = 256
    crop_size = 240
    flip = False
    serial_batch = False
    preprocess = 'resize'

    # brain
    # dataroot = '/Users/juntysun/Downloads/数据集/Brain_MRI_CT_Dataset/images'
    # dataroot = '/sunjindong/dataset/Brain'
    dataroot = '/sunjindong/dataset/npyFTTT'
    # dataroot = '/Users/jontysun/Downloads/数据集/BrainT1T2FT/npyFTTT'

    # model
    model = 'Pix2pixModel' # 'CycleGANModel'
    A = 't1'
    B = 't2'
    task = 'AtoB'
    in_channel = 1
    out_channel = 1

    # training
    save_dir = './ckpt'


