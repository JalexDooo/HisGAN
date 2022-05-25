import torch
import torch.nn as nn
import numpy as np
from models.base_model import ResnetBlock, BaseModel, get_norm_layer, init_net, UnetBlock, ADBlock, UpADBlock, ConvBlock
import functools
from distutils.version import LooseVersion
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import random

class ResnetGenerator(nn.Module):
    """
    ResnetGenerator(
        (model): Sequential(
        (0): ReflectionPad2d((3, 3, 3, 3))
        (1): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
        (2): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (3): ReLU(inplace=True)
        (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (5): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (8): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (9): ReLU(inplace=True)
        (10): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (11): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (12): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (13): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (14): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (15): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (16): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (17): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (18): ResnetBlock(
            (conv_block): Sequential(
            (0): ReflectionPad2d((1, 1, 1, 1))
            (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (2): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (3): ReLU(inplace=True)
            (4): ReflectionPad2d((1, 1, 1, 1))
            (5): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1))
            (6): InstanceNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            )
        )
        (19): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        (20): InstanceNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (21): ReLU(inplace=True)
        (22): ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1))
        (23): InstanceNorm2d(64, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
        (24): ReLU(inplace=True)
        (25): ReflectionPad2d((3, 3, 3, 3))
        (26): Conv2d(64, 3, kernel_size=(7, 7), stride=(1, 1))
        (27): Tanh()
        )
    )
    """
    def __init__(self, in_channel, out_channel, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        model = [nn.ReflectionPad2d(3), nn.Conv2d(in_channel, ngf, kernel_size=7, padding=0, bias=use_bias), norm_layer(ngf), nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3, stride=2, padding=1, bias=use_bias),norm_layer(ngf*mult*2), nn.ReLU(True)]
        
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf*mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling-i)
            model += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                    norm_layer(int(ngf*mult/2)),
                    nn.ReLU(True)
                    ]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, out_channel, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

class ADNet(nn.Module):
    def __init__(self, in_channel, out_channel, ngf=64, norm_layer=nn.BatchNorm2d):
        super(ADNet, self).__init__()
        self.encoder1 = ConvBlock(in_channel, ngf, kernel_size=3, stride=2, norm_layer=norm_layer)
        self.encoder2 = nn.Sequential(
            ADBlock(ngf, ngf*2, groups=16, stride=2, norm_layer=norm_layer),
            ADBlock(ngf*2, ngf*2, groups=16, stride=1, norm_layer=norm_layer),
            ADBlock(ngf*2, ngf*2, groups=16, stride=1, norm_layer=norm_layer)
        )
        self.encoder3 = nn.Sequential(
            ADBlock(ngf*2, ngf*4, groups=16, stride=2, norm_layer=norm_layer),
            ADBlock(ngf*4, ngf*4, groups=16, stride=1, norm_layer=norm_layer),
            ADBlock(ngf*4, ngf*4, groups=16, stride=1, norm_layer=norm_layer)
        )
        self.encoder4 = nn.Sequential(
            ADBlock(ngf*4, ngf*8, groups=16, stride=2, norm_layer=norm_layer),
            ADBlock(ngf*8, ngf*8, groups=16, stride=1, norm_layer=norm_layer),
            ADBlock(ngf*8, ngf*4, groups=16, stride=1, norm_layer=norm_layer)
        )
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up1 = nn.ConvTranspose2d(ngf*4, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.decoder1 = UpADBlock(ngf*4+ngf*4, ngf*4, groups=16, stride=1, norm_layer=norm_layer)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up2 = nn.ConvTranspose2d(ngf*4, ngf*4, kernel_size=4, stride=2, padding=1, bias=False)
        self.decoder2 = UpADBlock(ngf*4+ngf*2, ngf*2, groups=16, stride=1, norm_layer=norm_layer)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up3 = nn.ConvTranspose2d(ngf*2, ngf*2, kernel_size=4, stride=2, padding=1, bias=False)
        self.decoder3 = UpADBlock(ngf*2+ngf, ngf, groups=16, stride=1, norm_layer=norm_layer)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # self.up4 = nn.ConvTranspose2d(ngf, ngf, kernel_size=4, stride=2, padding=1, bias=False)
        self.decoder4 = ConvBlock(ngf, out_channel, kernel_size=1, stride=1, norm_layer=norm_layer)
        self.act = nn.Tanh() # The label pixel in range of [-1, 1]. So Tanh() will be fine.

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x = self.up1(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.decoder1(x)
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.decoder2(x)
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.decoder3(x)
        x = self.up4(x)
        x = self.decoder4(x)
        x = self.act(x)
        return x

class NLayerDiscriminator(nn.Module):
    """
    Gradient penalty needs NO Batch Normlization Layer.
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
                # norm_layer(ndf*nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            # norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        return self.model(x)

def define_G(in_channel, out_channel, ngf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = ADNet(in_channel, out_channel, ngf, norm_layer=norm_layer)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_EMA(in_channel, out_channel, ngf, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    # net = ResnetGenerator(in_channel, out_channel, ngf, norm_layer=norm_layer, n_blocks=9)
    net = ADNet(in_channel, out_channel, ngf, norm_layer=norm_layer)
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

class HistLoss(nn.Module):
    def __init__(self, bins=100):
        super().__init__()
        self.bins = bins
        self.loss = nn.L1Loss()
    
    def _min_max(self, img1, img2):
        self.minv = float(min(img1.min(), img2.min()))
        self.maxv = float(max(img1.max(), img2.max()))

    def _histc(self, img):
        if LooseVersion(torch.__version__) >= LooseVersion("1.10.0"):
            histc, bins = torch.histogram(img, bins=self.bins, range=(self.minv+0.1, self.maxv)) # for PyTorch>=1.10 version
            return histc, bins
        else:
            histc = torch.histc(img, bins=self.bins, min=self.minv+0.1, max=self.maxv)
            return histc, None
    
    def forward(self, prediction, target):
        self._min_max(prediction, target)
        histc_p, _ = self._histc(prediction.detach())
        histc_t, _ = self._histc(target.detach())
        loss = self.loss(histc_p, histc_t)
        return loss

def compute_gradient_penalty(Discriminator, real_sample, fake_sample):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    alpha = Tensor(np.random.random((real_sample.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_sample + (1 - alpha) * fake_sample).requires_grad_(True)
    d_interpolates = Discriminator(interpolates)
    grad_tensor = Variable(Tensor(d_interpolates.size(0), 1, d_interpolates.size(2), d_interpolates.size(3)).fill_(1.0), requires_grad=False)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs = grad_tensor,
        create_graph = True, # 设置为True可以计算更高阶的导数
        retain_graph = True, # 设置为True可以重复调用backward
        only_inputs = True, #默认为True，如果为True，则只会返回指定input的梯度值。 若为False，则会计算所有叶子节点的梯度，
                            #并且将计算得到的梯度累加到各自的.grad属性上去。
    )[0] # return a tensor list, get index 0.
    gradients = gradients.view(real_sample.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim = 1) - 1)**2).mean()
    return gradient_penalty

class KLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    def forward(self, prediction, target):
        prediction = prediction.view(prediction.size(0), -1)
        target = target.view(target.size(0), -1)
        prediction = F.log_softmax(prediction)
        target = F.log_softmax(target)
        loss = self.loss(prediction, target)
        return loss

class GDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pairwise_p_distance = torch.nn.PairwiseDistance(p=1.0)

    def forward(self, correct_images, generated_images):
        correct_images_gradient_x = self.calculate_x_gradient(correct_images)
        generated_images_gradient_x = self.calculate_x_gradient(generated_images)
        correct_images_gradient_y = self.calculate_y_gradient(correct_images)
        generated_images_gradient_y = self.calculate_y_gradient(generated_images)
        
        distances_x_gradient = self.pairwise_p_distance(
            correct_images_gradient_x, generated_images_gradient_x
        )
        distances_y_gradient = self.pairwise_p_distance(
            correct_images_gradient_y, generated_images_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss

    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                # [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            ]
        ).cuda()
        x_gradient_filter = x_gradient_filter.view(1, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, x_gradient_filter, groups=1, padding=(1, 1)
        )
        return result

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                # [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            ]
        ).cuda()
        y_gradient_filter = y_gradient_filter.view(1, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, y_gradient_filter, groups=1, padding=(1, 1)
        )
        return result

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
 
    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
 
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
 
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

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

class HisGAN_Baseline(BaseModel):
    """
        Note:
            `BaseLine` is a `no-cycle framework` by using L1+Adv+GDL for G and by using Adv+GP for D.

            The `G` Network is 2D model which improved based on DMFNet. (https://github.com/China-LiuXiaopeng/BraTS-DMFNet).

            The `D` Network applies no-normalization since the gradient penalty.
    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'G_L1', 'G_GDL', 'GP', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        if self.isTrain:
            self.netD = define_D(conf.in_channel, 64, 3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionKL = KLoss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.fake_B_pool = ImagePool(50)
            
            # self.emaD = EMA(self.netD, 0.999)
            # self.emaD.register()

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
        # fake_B = self.fake_B
        fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # compute gradient penalty
        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B.data, fake_B.data)*10.0
        # combine and backward
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP
        self.loss_D.backward()

    def backward_G(self):
        """loss for G"""
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10.0
        # self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * 100.0
        # self.loss_G_KL = self.criterionKL(self.fake_B, self.real_B) * 200.0
        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_GDL
        # + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # self.emaD.update()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()

class HisGAN_Baseline_Histloss(BaseModel):
    """
    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'G_L1', 'G_His', 'G_HisGDL', 'G_GDL', 'GP', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        self.emaG = EMA(self.netG, 0.9999) # No used.
        self.emaG.register()

        if self.isTrain:
            self.netD = define_D(conf.in_channel, 64, 3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionKL = KLoss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.alpha = [0.0]
            self.fake_B_pool = ImagePool(50)
            
            # self.emaD = EMA(self.netD, 0.999)
            # self.emaD.register()

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
        # fake_B = self.fake_B
        fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # compute gradient penalty
        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B.data, self.fake_B.data)*10.0
        # combine and backward
        self.loss_D = self.loss_D_fake + self.loss_D_real + self.loss_GP
        self.loss_D.backward()

    def backward_G(self):
        """loss for G"""
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10.0
        # self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * 100.0
        # self.loss_G_KL = self.criterionKL(self.fake_B, self.real_B) * 200.0
        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0
        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]

        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL
        # + self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # self.emaD.update()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.emaG.update()

class HisGAN_EMANet(BaseModel):
    """
        Note:

    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_total', 'G_GAN', 'G_L1', 'G_GDL', 'EMA_GAN', 'EMA_L1', 'D_real', 'D_fake', 'D_fake_AB', 'GP']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'random_AB', 'fake_AB']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        self.netEMA = define_EMA(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(conf.in_channel, 64, 3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionKL = KLoss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netEMA.parameters()), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.fake_B_pool = ImagePool(50)

            # self.emaD = EMA(self.netD, 0.999)
            # self.emaD.register()

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.real_B = input['B' if task else 'A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        alpha = Tensor(np.random.random((self.real_A.size(0), 1, 1, 1)))
        self.random_AB = (alpha * self.real_A + (1 - alpha) * self.fake_B).requires_grad_(True)
        self.fake_AB = self.netEMA(self.random_AB)

    def backward_D(self):
        """loss for D"""
        # fake
        # fake_B = self.fake_B
        fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_fake_AB = self.netD(self.fake_AB.detach())
        self.loss_D_fake_AB = self.criterionGAN(pred_fake_AB, False)

        # real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B.data, self.fake_B.data)*10.0
        # combine and backward
        self.loss_D = (self.loss_D_fake + self.loss_D_fake_AB)*0.5 + self.loss_D_real + self.loss_GP
        self.loss_D.backward()

    def backward_G(self):
        """loss for G"""
        fake_B = self.fake_B
        # fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10.0
        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0

        pred_fake_AB = self.netD(self.fake_AB)
        self.loss_EMA_GAN = self.criterionGAN(pred_fake_AB, True) * 0.1
        self.loss_EMA_L1 = self.criterionL1(self.fake_AB, self.real_B)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_GDL
        self.loss_EMA = self.loss_EMA_GAN + self.loss_EMA_L1
        self.loss_G_total = self.loss_G + self.loss_EMA
        # + self.loss_G_L1 
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # self.emaD.update()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

class HisGAN_EMANet_Histloss(BaseModel):
    """
        Note:
            with Norm
    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_total', 'G_GAN', 'G_L1', 'G_GDL', 'G_HisGDL', 'EMA_GAN', 'EMA_L1', 'D_real', 'D_fake', 'GP']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'random_AB', 'fake_AB']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        self.emaG = EMA(self.netG, 0.9999)
        self.emaG.register()

        self.netEMA = define_EMA(conf.in_channel, conf.out_channel, 64, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.netD = define_D(conf.in_channel, 64, 3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=self.gpu_ids)

            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss().to(self.device)
            self.criterionL2 = torch.nn.MSELoss().to(self.device)
            self.criterionHistc = HistLoss().to(self.device)
            self.criterionKL = KLoss().to(self.device)
            self.criterionGDL = GDLoss().to(self.device)
            
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netEMA.parameters()), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.alpha = [0.0]
            self.fake_B_pool = ImagePool(50)
            # fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images

            # self.emaD = EMA(self.netD, 0.999)
            # self.emaD.register()

    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.real_B = input['B' if task else 'A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']

    def forward(self):
        self.fake_B = self.netG(self.real_A)

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        alpha = Tensor(np.random.random((self.real_A.size(0), 1, 1, 1)))
        self.random_AB = (alpha * self.real_A + (1 - alpha) * self.fake_B).requires_grad_(True)
        self.fake_AB = self.netEMA(self.random_AB)

    def backward_D(self):
        """loss for D"""
        # fake
        # fake_B = self.fake_B
        fake_B = self.fake_B_pool.query(self.fake_B) # for unpaired images
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        pred_fake_AB = self.netD(self.fake_AB.detach())
        self.loss_D_fake_AB = self.criterionGAN(pred_fake_AB, False)

        # real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_GP = compute_gradient_penalty(self.netD, self.real_B.data, self.fake_B.data)*10.0
        # combine and backward
        self.loss_D = (self.loss_D_fake + self.loss_D_fake_AB)*0.5 + self.loss_D_real + self.loss_GP
        self.loss_D.backward()

    def backward_G(self):
        """loss for G"""
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10.0
        self.loss_G_GDL = self.criterionGDL(self.fake_B, self.real_B) * 10.0

        pred_fake_AB = self.netD(self.fake_AB)
        self.loss_EMA_GAN = self.criterionGAN(pred_fake_AB, True) * 0.5
        self.loss_EMA_L1 = self.criterionL1(self.fake_AB, self.real_B) * 10.0 * 0.5

        self.loss_G_His = self.criterionHistc(self.fake_B, self.real_B) /100.0
        self.alpha.append(self.loss_G_His)
        alpha = torch.Tensor(self.alpha)
        alpha = (alpha - alpha.mean())/alpha.std()
        alpha = torch.sigmoid(alpha)[-1]
        self.loss_G_HisGDL = self.loss_G_GDL * float(1.0+alpha)

        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_HisGDL
        self.loss_EMA = self.loss_EMA_GAN + self.loss_EMA_L1
        self.loss_G_total = self.loss_G + self.loss_EMA
        # + self.loss_G_L1 
        self.loss_G_total.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        # self.emaD.update()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

