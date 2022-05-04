import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel, get_norm_layer, init_net

class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu):
        super(UNetConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn(self.conv(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3, activation=F.relu, space_dropout=False):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, 2, stride=2)
        self.bnup = nn.BatchNorm2d(out_size)
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_size)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.activation = activation

    def center_crop(self, layer, target_size):
        batch_size, n_channels, layer_width, layer_height = layer.size()
        xy1 = (layer_width - target_size) // 2
        return layer[:, :, xy1:(xy1 + target_size), xy1:(xy1 + target_size)]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.activation(self.bnup(up))
        crop1 = self.center_crop(bridge, up.size()[2])
        out = torch.cat([up, crop1], 1)

        out = self.activation(self.bn(self.conv(out)))
        out = self.activation(self.bn2(self.conv2(out)))

        return out

class UNet_LRes(nn.Module):
    def __init__(self, in_channel = 1, n_classes = 1):
        super(UNet_LRes, self).__init__()
#         self.imsize = imsize

        self.activation = F.relu
        
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        self.conv_block1_64 = UNetConvBlock(in_channel, 64)
        self.conv_block64_128 = UNetConvBlock(64, 128)
        self.conv_block128_256 = UNetConvBlock(128, 256)
        self.conv_block256_512 = UNetConvBlock(256, 512)
        self.conv_block512_1024 = UNetConvBlock(512, 1024)
        # this kind of symmetric design is awesome, it automatically solves the number of channels during upsamping
        self.up_block1024_512 = UNetUpBlock(1024, 512)
        self.up_block512_256 = UNetUpBlock(512, 256)
        self.up_block256_128 = UNetUpBlock(256, 128)
        self.up_block128_64 = UNetUpBlock(128, 64)

        self.last = nn.Conv2d(64, n_classes, 1, stride=1)


    def forward(self, x):
        res_x = x
#         print 'line 70 ',x.size()
        block1 = self.conv_block1_64(x)
        pool1 = self.pool1(block1)

        block2 = self.conv_block64_128(pool1)
        pool2 = self.pool2(block2)

        block3 = self.conv_block128_256(pool2)
        pool3 = self.pool3(block3)

        block4 = self.conv_block256_512(pool3)
        pool4 = self.pool4(block4)

        block5 = self.conv_block512_1024(pool4)

        up1 = self.up_block1024_512(block5, block4)

        up2 = self.up_block512_256(up1, block3)

        up3 = self.up_block256_128(up2, block2)

        up4 = self.up_block128_64(up3, block1)
        
        last = self.last(up4)
        #print 'res_x.shape is ',res_x.shape,' and last.shape is ',last.shape
        if len(res_x.shape)==3:
            res_x = res_x.unsqueeze(1) 
        out = torch.add(last,res_x)
        
        #print 'out.shape is ',out.shape
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel, ndf=32):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, ndf, kernel_size=9),
            nn.ReLU(True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf*2, kernel_size=5),
            nn.ReLU(True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf*2, ndf*2, kernel_size=5),
            nn.ReLU(True)
        )
        self.fc1 = nn.Linear(28*28*64, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), (2, 2))
        x = F.max_pool2d(self.conv2(x), (2, 2))
        x = F.max_pool2d(self.conv3(x), (2, 2))
        # print(x.shape)
        x = x.view(x.size(0), -1) # 50176
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

def define_G(in_channel, out_channel, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = UNet_LRes(in_channel, out_channel)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(in_channel, ndf, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = Discriminator(in_channel, ndf)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.BCELoss()
    
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

class GDLoss(nn.Module):
    def __init__(self, pNorm=2):
        super().__init__()
        self.convX = nn.Conv2d(1, 1, kernel_size=(1, 2), stride=1, padding=(0, 1), bias=False)
        self.convY = nn.Conv2d(1, 1, kernel_size=(2, 1), stride=1, padding=(1, 0), bias=False)
        filterX = torch.FloatTensor([[[[-1, 1]]]])  # [1, 1, 1, 2]
        filterY = torch.FloatTensor([[[[1], [-1]]]])  # [1, 1, 2, 1]
        self.convX.weight = nn.Parameter(filterX, requires_grad=False)
        self.convY.weight = nn.Parameter(filterY, requires_grad=False)
        self.pNorm = pNorm
    
    def forward(self, pred, target):
        pred_dx = torch.abs(self.convX(pred))
        pred_dy = torch.abs(self.convY(pred))
        target_dx = torch.abs(self.convX(target))
        target_dy = torch.abs(self.convY(target))
        grad_diff_x = torch.abs(target_dx - pred_dx)
        grad_diff_y = torch.abs(target_dy - pred_dy)
        mat_loss_x = grad_diff_x ** self.pNorm
        mat_loss_y = grad_diff_y ** self.pNorm
        shape = target.shape
        mean_loss = (torch.sum(mat_loss_x) + torch.sum(mat_loss_y)) / (shape[0] * shape[1] * shape[2] * shape[3])
        return mean_loss


class MedGANModel(BaseModel):
    """
    https://github.com/ginobilinie/medSynthesisV1
    Some debugs to reproduce. 2D
    """
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GDL', 'G_ADV', 'G_L2', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        
        self.netG = define_G(conf.in_channel, conf.out_channel, init_type='xavier', init_gain=np.sqrt(2.0), gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = define_D(conf.in_channel+conf.out_channel, ndf=32, init_type='xavier', init_gain=np.sqrt(2.0), gpu_ids=self.gpu_ids)
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL2 = nn.MSELoss().to(self.device)
            self.gdl = GDLoss().to(self.device)

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
        self.loss_G_ADV = self.criterionGAN(pred_fake, True) * 0.5
        self.loss_G_L2 = self.criterionL2(self.fake_B, self.real_B) * 1.0
        self.loss_G_GDL = self.gdl(self.fake_B, self.real_B) * 1.0


        self.loss_G = self.loss_G_ADV + self.loss_G_L2 + self.loss_G_GDL
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
