import torch
import torch.nn as nn
from models.base_model import BaseModel, get_norm_layer, init_net, UnetBlock
import functools

class UnetGenerator(nn.Module):
    def __init__(self, in_channel, out_channel, num_downsample, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()
        unet_block = UnetBlock(None, ngf*8, ngf*8, pre_module=None, norm_layer=norm_layer, use_dropout=use_dropout, inner=True)
        for _ in range(num_downsample-5):
            unet_block = UnetBlock(None, ngf*8, ngf*8, pre_module=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetBlock(None, ngf*4, ngf*8, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf*2, ngf*4, pre_module=unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock(None, ngf, ngf*2, pre_module=unet_block, norm_layer=norm_layer)
        self.model = UnetBlock(in_channel, out_channel, ngf, pre_module=unet_block, outer=True, norm_layer=norm_layer)
    
    def forward(self, x):
        return self.model(x)

class NLayerDiscriminator(nn.Module):
    """
    NLayerDiscriminator(
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

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

def define_G(in_channel, out_channel, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = UnetGenerator(in_channel, out_channel, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(in_channel, ndf, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(in_channel, ndf, n_layers=n_layers_D, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCEWithLogitsLoss()
    
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

class Pix2pixModel(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = define_D(conf.in_channel+conf.out_channel, 64, 3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        
        if self.isTrain:
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
    
    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.real_B = input['B' if task else 'A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
    
    def forward(self):
        self.fake_B = self.netG(self.real_A)
    
    def backward_D(self):
        """loss for D"""
        # fake
        fake_AB = torch.cat([self.real_A, self.fake_B], dim=1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # real
        real_AB = torch.cat([self.real_A, self.real_B], dim=1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine and backward
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        """loss for G"""
        fake_AB = torch.cat([self.real_A, self.fake_B], dim=1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 100.0
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
