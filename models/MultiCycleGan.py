"""
MultiCycleGAN
"""

import torch
import torch.nn as nn
from .base_model import BaseModel, get_norm_layer, init_net, UnetBlock_with_z, D_NLayersMulti, E_ResNet


class G_Unet_add_all(nn.Module):
    def __init__(self, in_channel, out_channel, nz, num_downs, ngf=64, norm_layer=None, use_dropout=False):
        super(G_Unet_add_all, self).__init__()
        self.nz = nz
        unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, None, inner=True, norm_layer=norm_layer)
        unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, unet_block, use_dropout=use_dropout, norm_layer=norm_layer)
        for _ in range(num_downs-6):
            unet_block = UnetBlock_with_z(ngf*8, ngf*8, ngf*8, nz, unet_block, use_dropout=use_dropout, norm_layer=norm_layer)
        unet_block = UnetBlock_with_z(ngf*4, ngf*4, ngf*8, nz, unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock_with_z(ngf*2, ngf*2, ngf*4, nz, unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock_with_z(ngf, ngf, ngf*2, nz, unet_block, norm_layer=norm_layer)
        unet_block = UnetBlock_with_z(in_channel, out_channel, ngf, nz, unet_block, outer=True, norm_layer=norm_layer)
        self.model = unet_block
    
    def forward(self, x, z):
        return self.model(x, z)

def define_G(in_channel, out_channel, nz=8, ngf=64, norm='batch', use_dropout=False, init_type='xavier', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = G_Unet_add_all(in_channel, out_channel, nz, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_D(in_channel, ndf, norm='batch', init_type='xavier', init_gain=0.02, num_D=1, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    net = D_NLayersMulti(in_channel, ndf, 3, norm_layer=norm_layer, num_D=num_D)
    return init_net(net, init_type, init_gain, gpu_ids)

def define_E(in_channel, out_channel, ndf, norm='batch', init_type='xavier', init_gain=0.02, gpu_ids=[], vaeLike=False):
    norm_layer = get_norm_layer(norm)
    net = E_ResNet(in_channel, out_channel, ndf, n_blocks=5, norm_layer=norm_layer, vaeLike=vaeLike)
    return init_net(net, init_type, init_gain, gpu_ids)

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.loss = nn.MSELoss()
    
    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, predictions, target_is_real):
        all_losses = []
        for prediction in predictions:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
            all_losses.append(loss)
        total_loss = sum(all_losses)
        return total_loss, all_losses


class MultiCycleGANModel(BaseModel):
    def __init__(self, conf):
        BaseModel.__init__(self, conf)
        self.loss_names = ['G_GAN', 'D', 'G_GAN2', 'D2', 'G_L1', 'z_L1', 'kl']
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']
        self.model_names = ['G']

        use_D = True if self.isTrain else False
        use_D2 = False
        use_E = True # if self.isTrain else False
        use_vae = True

        self.netG = define_G(conf.in_channel, conf.out_channel, nz=8, ngf=64, norm='instance', use_dropout=True, init_type='xavier', init_gain=0.02, gpu_ids=self.gpu_ids)

        D_output_nc = conf.in_channel + conf.out_channel

        if use_D:
            self.model_names += ['D']
            self.netD = define_D(D_output_nc, ndf=64, norm='instance', num_D=2, init_type='xavier', init_gain=0.02, gpu_ids=self.gpu_ids)
        
        if use_D2:
            self.model_names += ['D2']
            self.netD2 = define_D(D_output_nc, ndf=64, norm='instance', num_D=2, init_type='xavier', init_gain=0.02, gpu_ids=self.gpu_ids)
        else:
            self.netD2 = None
        
        if use_E:
            self.model_names += ['E']
            self.netE = define_E(conf.out_channel, 8, 64, norm='instance', init_type='xavier', init_gain=0.02, gpu_ids=self.gpu_ids, vaeLike=use_vae)
        

        if self.isTrain: # lsgan
            self.criterionGAN = GANLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionZ = torch.nn.L1Loss()

            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=conf.lr, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            if use_E:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(), lr=conf.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_E)
            if use_D:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=conf.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_D)
            if use_D2:
                self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(), lr=conf.lr, betas=(0.5, 0.999))
                self.optimizers.append(self.optimizer_D2)
    
    def set_input(self, input):
        task = self.conf.task == 'AtoB'
        self.real_A = input['A' if task else 'B'].to(self.device)
        self.real_B = input['B' if task else 'A'].to(self.device)
        self.image_paths = input['A_paths' if task else 'B_paths']
    
    def get_z_random(self, batch_size, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(batch_size, nz)*2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batch_size, nz)
        return z.detach().to(self.device)
    
    def encode(self, x):
        mu, logvar = self.netE.forward(x)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar
    
    def forward(self):
        half_size = self.conf.batch_size // 2
        self.real_A_encoded = self.real_A[0:half_size]
        self.real_B_encoded = self.real_B[0:half_size]
        self.real_A_random = self.real_A[half_size:]
        self.real_B_random = self.real_B[half_size:]

        self.z_encoded, self.mu, self.logvar = self.encode(self.real_B_encoded)
        self.z_random = self.get_z_random(self.real_A_encoded.size(0), nz=8)
        
        self.fake_B_encoded = self.netG(self.real_A_encoded, self.z_encoded)
        self.fake_B_random = self.netG(self.real_A_encoded, self.z_random)

        self.fake_data_encoded = torch.cat([self.real_A_encoded, self.fake_B_encoded], 1)
        self.real_data_encoded = torch.cat([self.real_A_encoded, self.real_B_encoded], 1)
        self.fake_data_random = torch.cat([self.real_A_encoded, self.fake_B_random], 1)
        self.real_data_random = torch.cat([self.real_A_random, self.real_B_random], 1)

        self.mu2, logvar2 = self.netE(self.fake_B_random)
    
    def backward_D(self, netD, real, fake):
        pred_fake = netD(fake.detach())
        pred_real = netD(real)
        loss_D_fake, _ = self.criterionGAN(pred_fake, False)
        loss_D_real, _ = self.criterionGAN(pred_real, True)
        loss_D = loss_D_fake + loss_D_real
        loss_D.backward()
        return loss_D, [loss_D_fake, loss_D_real]
    
    def backward_G_GAN(self, fake, netD=None, ll=0.0):
        if ll > 0.0:
            pred_fake = netD(fake)
            loss_G_GAN, _ = self.criterionGAN(pred_fake, True)
        else:
            loss_G_GAN = 0
        return loss_G_GAN * ll
    
    def backward_EG(self):
        # 1, G(A) should fool D
        self.loss_G_GAN = self.backward_G_GAN(self.fake_data_encoded, self.netD, 1.0)
        self.loss_G_GAN2 = self.backward_G_GAN(self.fake_data_random, self.netD, 1.0)
        # 2. KL loss
        self.loss_kl = torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp()) * (-0.5 * 0.01)
        # 3, reconstruction |fake_B-real_B|
        self.loss_G_L1 = self.criterionL1(self.fake_B_encoded, self.real_B_encoded) * 10.0

        self.loss_G = self.loss_G_GAN + self.loss_G_GAN2 + self.loss_G_L1 + self.loss_kl
        self.loss_G.backward(retain_graph=True)

    def update_D(self):
        self.set_requires_grad([self.netD, self.netD2], True)
        # update D1
        self.optimizer_D.zero_grad()
        self.loss_D, self.losses_D = self.backward_D(self.netD, self.real_data_encoded, self.fake_data_encoded)
        self.loss_D2, self.losses_D2 = self.backward_D(self.netD, self.real_data_random, self.fake_data_random)
        self.optimizer_D.step()

    def backward_G_alone(self):
        # 3, reconstruction |(E(G(A, z_random)))-z_random|
        self.loss_z_L1 = self.criterionZ(self.mu2, self.z_random) * 0.5
        self.loss_z_L1.backward()

    def update_G_and_E(self):
        # update G and E
        self.set_requires_grad([self.netD, self.netD2], False)
        self.optimizer_E.zero_grad()
        self.optimizer_G.zero_grad()
        self.backward_EG()

        # update G alone
        self.set_requires_grad([self.netE], False)
        self.backward_G_alone()
        self.set_requires_grad([self.netE], True)

        self.optimizer_E.step()
        self.optimizer_G.step()


    def optimize_parameters(self):
        self.forward()
        self.update_G_and_E()
        self.update_D()



