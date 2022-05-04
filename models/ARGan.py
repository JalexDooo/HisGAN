import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel, get_norm_layer, init_net, UnetBlock
import functools
import torch.fft as fft
import random

"""
Requires PyTorch >= 1.8.1 due to torch.fft.fft
"""

class AR_ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_layer=nn.BatchNorm2d):
        super(AR_ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, 1, 1),
            norm_layer(out_channel),
            nn.RReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class ARNetGenerator(nn.Module):
    """ As the paper said.
    ``Specifically, the AR-Net contains eight Encoder/Decoder blocks with the Conv + BN + Leaky-ReLU/ReLU components, in which all the convolution layers apply 3*3 kernels with stride of 1. 
    In the encoder part, we utilize two 2 * 2 max- pooling layers with stride of 2 for down-sampling the input image. 
    Similarly, transposed convolutions with a 2 * 2 filter of stride 2 is also employed in the decoder part for up-sampling.``

    Note that, the skip connection strategy is not utilized in AR-Net, since skip connection mainly aims to retain the low-level information of the input image while our AR-Net focuses on high-level information protection.

    Note:
        The same setting with the above.
    """
    def __init__(self, in_channel, out_channel, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ARNetGenerator, self).__init__()
        self.ec1 = AR_ConvBlock(in_channel, ngf, norm_layer)
        self.ec2 = AR_ConvBlock(ngf, ngf*2, norm_layer)
        self.ec3 = AR_ConvBlock(ngf*2, ngf*4, norm_layer)
        self.ec4 = AR_ConvBlock(ngf*4, ngf*8, norm_layer)

        self.dc1 = AR_ConvBlock(ngf*8, ngf*4, norm_layer)
        self.tc1 = nn.ConvTranspose2d(ngf*4, ngf*4, kernel_size=2, stride=2, padding=0, bias=False)
        self.dc2 = AR_ConvBlock(ngf*4, ngf*2, norm_layer)
        self.tc2 = nn.ConvTranspose2d(ngf*2, ngf*2, kernel_size=2, stride=2, padding=0, bias=False)
        self.dc3 = AR_ConvBlock(ngf*2, ngf, norm_layer)

        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.outer = nn.Sequential(
            nn.Conv2d(ngf, out_channel, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        res = x
        x = self.ec1(x)
        x = self.maxpool1(x)
        x = self.ec2(x)
        x = self.maxpool2(x)
        x = self.ec3(x)
        x = self.ec4(x)

        x = self.dc1(x)
        x = self.tc1(x)
        x = self.dc2(x)
        x = self.tc2(x)
        x = self.dc3(x)
        x = self.outer(x)
        return x*res

class PreNetGenerator(nn.Module):
    """ As the paper said.
    ``Concretely, the encoder contains seven convolution blocks (Conv Block) while the decoder comprises the same number of deconvolution blocks (de-Conv Block). 
    Both of them adopt a 4 * 4 convolution kernel with stride 2 for convolution or transposed convolution, so that the size of the feature map is reduced or increased by multiples in the encoder and decoder, respectively.``

    Note:
        The most deep channel set to 256. The paper is not given.
    """
    def __init__(self, in_channel, out_channel, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PreNetGenerator, self).__init__()
        unet_block = UnetBlock(None, ngf*8, ngf*8, pre_module=None, norm_layer=norm_layer, use_dropout=use_dropout, inner=True)
        unet_block = UnetBlock(None, ngf*8, ngf*8, pre_module=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(None, ngf*8, ngf*8, pre_module=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(None, ngf*4, ngf*8, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf*2, ngf*4, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf, ngf*2, pre_module=unet_block, norm_layer=norm_layer)
        self.pre_model = UnetBlock(in_channel, out_channel, ngf, pre_module=unet_block, outer=True, norm_layer=norm_layer)

        self.ar_model = ARNetGenerator(in_channel, out_channel, ngf=ngf, norm_layer=norm_layer)

        self.act = nn.Sigmoid()
    
    def forward(self, x):
        Pre = self.pre_model(x)

        Ar = self.ar_model(Pre)

        return Pre, Ar, self.act(Pre+Ar)

class AdvDiscriminator(nn.Module):
    """
    Notes:
        No more details. Implements of Pix2pix discriminator.  ``Isola et al. (2017). paper``

    AdvDiscriminator(
        (model): Sequential(
            (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (1): LeakyReLU(negative_slope=0.2, inplace=True)
            (2): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (3): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (4): LeakyReLU(negative_slope=0.2, inplace=True)
            (5): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (7): LeakyReLU(negative_slope=0.2, inplace=True)
            (8): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
            (9): InstanceNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (10): LeakyReLU(negative_slope=0.2, inplace=True)
            (11): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
        )
    )
    """
    def __init__(self, in_channel, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(in_channel, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf*nf_mult_prev, ndf*nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), nn.Tanh()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

def define_G(in_channel, out_channel, ngf=64, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = PreNetGenerator(in_channel, out_channel, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(in_channel, ndf=64, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = AdvDiscriminator(in_channel, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

class AdvLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.loss(prediction, target_tensor)
        return loss

class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.Sigmoid()
    
    def __call__(self, rectified, target):
        target = self.act(target)
        fft_rectified = fft.fftn(rectified, dim=(-2, -1))
        fft_rectified = fft.fftshift(fft_rectified)
        fft_rectified = torch.log(torch.abs(fft_rectified)+1) # +1 避免可能的log计算负值为nan / +1 to avoid log() -> nan
        fft_target = fft.fftn(target, dim=(-2, -1))
        fft_target = fft.fftshift(fft_target)
        fft_target = torch.log(torch.abs(fft_target)+1) 
        fft_rectified = self.act(fft_rectified)
        fft_target = self.act(fft_target)
        loss = self.loss(fft_rectified, fft_target)
        return loss

class ImagePool():
    """This class implments an image buffer that stores previously generated images.
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs +1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size -1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

class ARGANModel(BaseModel):
    """
    ``fake_B``: Rectified prediction
    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)

        self.loss_names = ['G_P_L1', 'G_R_L1', 'G_S', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']

        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(conf.in_channel+conf.out_channel, ndf=64, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
            self.criterionL1 = nn.L1Loss().to(self.device)
            self.criterionAdv = AdvLoss().to(self.device)
            self.crierionSpectral = SpectralLoss().to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)

    
    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.real_B = input['B' if task else 'A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']

    def forward(self):
        self.fake_B, self.res, self.rectified = self.netG(self.real_A)
        if self.isTrain:
            self.res_gt = self.real_B - self.real_A

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        fake_AB = torch.cat([self.real_A, fake_B], dim=1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionAdv(pred_fake, False)

        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionAdv(pred_real, True)
        # combine and backward
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat([self.real_A, self.fake_B], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_Adv = self.criterionAdv(pred_fake, True)

        self.loss_G_P_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_R_L1 = self.criterionL1(self.res, self.res_gt)
        self.loss_G_S = self.crierionSpectral(self.rectified, self.real_B)

        self.loss_G = self.loss_G_Adv + self.loss_G_P_L1*100.0 + self.loss_G_R_L1*10.0 + self.loss_G_S*10.0
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G_PreNet
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


