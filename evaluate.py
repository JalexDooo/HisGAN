import torch
import numpy as np
import glob
import math
import models
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

def tableResult():
    alls = glob.glob('./output/*')
    slc = 77
    dataset = list(range(1, 126))
    rem = [67, 68, 69, 72, 74, 76, 77, 83, 85, 89, 91, 92, 99, 103]
    for i in rem:
        dataset.remove(i)

    # PSNR
    def PSNR(fake, real, mse):
        if mse < 1.0e-10:
            return 100
        return 20.0 * math.log10(1.0/mse)

    # MSE
    MSE = torch.nn.MSELoss()

    # SSIM
    from skimage.metrics import structural_similarity as SSIM

    for idx in alls:
        file = idx.split('/')[-1]
        if 'MR' in file:
            continue
        if 'flair' in file:
            name = file[:-14]
            task = 'FlairtoT1ce'
        if 't2' in file:
            name = file[:-9]
            task = 'T1toT2'
        if name != 'HisGAN_EMANet_Histloss':
            continue
        print(name, task)
        mses = []
        psnrs = []
        ssims = []

        for i in dataset:
            real_A_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_real_A.npy'.format(i, slc)
            real_B_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_real_B.npy'.format(i, slc)
            fake_B_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_fake_B.npy'.format(i, slc)
            real_A = np.load(real_A_p)
            real_B = np.load(real_B_p)
            fake_B = np.load(fake_B_p)
            real_A = torch.from_numpy(real_A)
            real_B = torch.from_numpy(real_B)
            fake_B = torch.from_numpy(fake_B)

            mse = float(MSE(fake_B, real_B))
            psnr = float(PSNR(fake_B, real_B, mse))
            ssim = float(SSIM(fake_B.numpy(), real_B.numpy()))

            mses.append(mse)
            psnrs.append(psnr)
            ssims.append(ssim)
        mses = np.array(mses)
        psnrs = np.array(psnrs)
        ssims = np.array(ssims)
        print('Task: {}, Model: {} ----> MSE:{}({}), PSNR:{}({}), SSIM:{}({})'.format(
            task, name, mses.mean(), mses.std(), psnrs.mean(), psnrs.std(), ssims.mean(), ssims.std()
        ))
        # break

def MRtoCT():
    from skimage.metrics import structural_similarity as SSIM

    def Avg_Log_likeerror(paths1):
        All1 = 0
        for i, npy in enumerate(paths1):
            img1 = np.load(npy)
            if not i:
                mean_img = img1
            else:
                mean_img += img1
        mean_img = mean_img/(i+1)
        for i, npy in enumerate(paths1):
            img1 = np.load(npy)
            All1 += np.log(np.abs(mean_img-img1).mean())
        All1 = All1/(i+1)
        return All1
    
    def Wasserstein_dis(paths1, paths2, task):
        discriminator = models.ResCycleGan.NLayerDiscriminator(in_channel=1, norm_layer=torch.nn.InstanceNorm2d)
        if isinstance(discriminator, torch.nn.DataParallel):
            discriminator = discriminator.module
        if task == 'MRtoCT':
            state_dict = torch.load('ckpt/AtoB(MRtoCT)CycleGANModel/iter_20_net_D_A.pth', map_location='cpu')
        elif task =='T1toT2':
            state_dict = torch.load('ckpt/AtoB(T1toT2)CycleGANModel/latest_net_D_A.pth', map_location='cpu')
        elif task == 'FlairtoT1ce':
            state_dict = torch.load('ckpt/AtoB(FlairtoT1ce)CycleGANModel/latest_net_D_A.pth', map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in state_dict.items(): # As for ckpt is gpu-parallel model. To CPU
            name = k[7:]
            new_state_dict[name] = v
        discriminator.load_state_dict(new_state_dict)

        imgs1 = []
        for i, p1 in enumerate(paths1):
            img = np.load(p1)
            img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
            o1 = discriminator(img).squeeze(0).squeeze(0)
            imgs1.append(o1.detach().numpy())
        imgs1 = np.array(imgs1)
        imgs1 = np.sum(imgs1, axis=0)/(i+1)

        imgs2 = []
        for i, p2 in enumerate(paths2):
            img = np.load(p2)
            img = torch.Tensor(img).unsqueeze(0).unsqueeze(0)
            o2 = discriminator(img).squeeze(0).squeeze(0)
            imgs2.append(o2.detach().numpy())
        imgs2 = np.array(imgs2)
        imgs2 = np.sum(imgs2, axis=0)/(i+1)

        error = np.abs(imgs1 - imgs2)
        return error.mean(), error.std()

    def compute_SSIM(paths1, paths2):
        ssims = []
        for i in range(max(len(paths1), len(paths2))):
            path1 = paths1[i%(len(paths1))]
            path2 = paths2[i%(len(paths2))]
            img1 = np.load(path1)
            img2 = np.load(path2)
            ssims.append(float(SSIM(img1, img2)))
        ssims = np.array(ssims)
        return ssims.mean(), ssims.std()
        
            
    alls = glob.glob('./output/*')
    for i in alls:
        model = i.split('/')[-1][:-9]
        task = 'MRtoCT'
        if 'CT' not in i:
            continue
        if 'flair' in i:
            model = i.split('/')[-1][:-14]
            task = 'FlairtoT1ce'
        if 't2' in i:
            model = i.split('/')[-1][:-9]
            task = 'T1toT2'

        fake_npys = glob.glob(i+'/npy/*fake_B.npy')
        real_npys = glob.glob(i+'/npy/*real_B.npy')

        average_log_likehood = np.abs(Avg_Log_likehood(fake_npys) - Avg_Log_likehood(real_npys))
        mean_W, std_W = Wasserstein_dis(fake_npys, real_npys, task)

        mean_SSIM, std_SSIM = compute_SSIM(fake_npys, real_npys)
        print('Task: {},  Model: {},  ALL: {},  WD: {}({}),  SSIM:  {}({})'.format(task, model, average_log_likehood, mean_W, std_W, mean_SSIM, std_SSIM))
        # assert False

"""
Flair->T1ce:
Model: CycleGANModel ----> MSE:0.025623436836811068(0.024238823353049173), PSNR:34.280839499601946(6.092347054548182), SSIM:0.7597502807831493(0.06342704336183304)
Model: Pix2pixModel ----> MSE:0.02162533107020699(0.024283333854110026), PSNR:36.24831278868342(6.437901686520218), SSIM:0.7786936523659147(0.07194680180655863)
Model: BiCycleGANModel ----> MSE:0.024155060102038824(0.027118993171551024), PSNR:35.00563469971977(5.928130835001194), SSIM:0.7664734595727912(0.06547870053261637)
Model: MedGANModel ----> MSE:0.01985673682327877(0.01983129334935044), PSNR:36.43450502098144(5.813704634062967), SSIM:0.7732278491742597(0.06367013063018027)
Model: ARGANModel ----> MSE:0.022097547379162936(0.022864329178726387), PSNR:36.062094761942596(6.6680781250620775), SSIM:0.7775475812973077(0.0730263418469996)
Model: HisGAN_Baseline ----> MSE:0.017930618004856614(0.0185543934924078), PSNR:37.91432103737383(6.716930411039956), SSIM:0.8070549559123221(0.08678926533883986)
Model: HisGAN_Baseline_Histloss ----> MSE:0.01850288623717387(0.020506166150972446), PSNR:37.81620605128687(6.855256865165234), SSIM:0.8068910639230339(0.08960092363785661)
Model: HisGAN_EMANet ----> MSE:0.018322239294312558(0.020012618162476115), PSNR:37.81363750029924(6.725040672365062), SSIM:0.8064139711239384(0.08829757277124237)
Model: HisGAN_EMANet_Histloss ----> MSE:0.018835699508755323(0.020868767647585277), PSNR:37.58085848882372(6.678775234763067), SSIM:0.8033512307611469(0.087778693734146)

T1->T2:
Model: CycleGANModel ----> MSE:0.029865444117636832(0.028967144102972263), PSNR:32.89153374932408(5.880378583486296), SSIM:0.7933304349614212(0.06209981516175062)
Model: Pix2pixModel ----> MSE:0.021997398621327168(0.0289026467910059), PSNR:36.61524433408616(6.773982659991025), SSIM:0.8322291292239367(0.072283946254203)
Model: BiCycleGANModel ----> MSE:0.023957213219087403(0.027110159249138257), PSNR:35.23036558576932(6.287684486006726), SSIM:0.8160475137927031(0.06749320366454942)
Model: MedGANModel ----> MSE:0.020305998145124397(0.023521426002525176), PSNR:36.70055998165267(6.1323035566097435), SSIM:0.8254308132746587(0.06673827829372037)
Model: ARGANModel ----> MSE:0.02080198785139097(0.025404340338901505), PSNR:36.69712803856869(6.39206366564696), SSIM:0.8318701523925487(0.07073190613741868)
Model: HisGAN_Baseline ----> MSE:0.019051375626880036(0.02356669788243081), PSNR:37.61078212593778(6.471926636847989), SSIM:0.8500362604046703(0.07599150249694374)
Model: HisGAN_Baseline_Histloss ----> MSE:0.01895163130216502(0.02441692083014266), PSNR:37.86025214747084(6.670800667432189), SSIM:0.8517109386907638(0.0773559353706142)
Model: HisGAN_EMANet ----> MSE:0.018634817726608063(0.023757134996049204), PSNR:37.90876266025892(6.5938907951423404), SSIM:0.8524333928797723(0.07666033296755172)
Model: HisGAN_EMANet_Histloss ----> MSE:0.018832431938331406(0.024314901178966845), PSNR:37.82645682906723(6.5473538825999755), SSIM:0.8515012290079277(0.07707166292783554)

MR->CT:
Model: CycleGANModel,  ALL: 0.22730863889058428,  WD: 0.3976287841796875(0.0796232745051384),  SSIM:  0.5769366435737628
Model: Pix2pix,  ALL: 0.19265081087748204,  WD: 0.43515753746032715(0.07483110576868057),  SSIM:  0.5771495018879176
Model: BiCycleGANModel,  ALL: 0.17743165890375767,  WD: 0.4900749921798706(0.07571025937795639),  SSIM:  0.5828751215725251
Model: MedGANModel,  ALL: 0.19311518271764117,  WD: 0.37323981523513794(0.08179966360330582),  SSIM:  0.5717347096703314
Model: ARGANModel,  ALL: 0.2104256629943848,  WD: 0.2656009793281555(0.07945609092712402),  SSIM:  0.5753037405313375
Model: HisGAN_Base,  ALL: 0.1880606889724732,  WD: 0.33732926845550537(0.08286960422992706),  SSIM:  0.561128983825204888
Model: HisGAN_Base_His,  ALL: 0.1714149316151936,  WD: 0.23247073590755463(0.07651521265506744),  SSIM:  0.5745342217749481
Model: HisGAN_EMA,  ALL: 0.18251893122990914,  WD: 0.2640434503555298(0.07974710315465927),  SSIM:  0.5681766471529601
Model: HisGAN_EMA_His,  ALL: 0.19071842034657793,  WD: 0.22319164872169495(0.07673940062522888),  SSIM:  0.5766757607108458


"""

def PSNR(fake, real, mse):
    if mse < 1.0e-10:
        return 100
    return 20.0 * math.log10(1.0/mse)

def figureResult():
    alls = glob.glob('./output/*')
    slc = 77 # slice number
    dataset = list(range(1, 126))
    rem = [67, 68, 69, 72, 74, 76, 77, 83, 85, 89, 91, 92, 99, 103] # The validation set has 111 Images.
    for i in rem:
        dataset.remove(i)

    # MSE
    MSE = torch.nn.MSELoss()

    # SSIM
    from skimage.metrics import structural_similarity as SSIM


    for idx in alls:
        file = idx.split('/')[-1]
        if 'MR' in file:
            continue
        if 'flair' in file:
            name = file[:-14]
            task = 'FlairtoT1ce'
        if 't2' in file:
            name = file[:-9]
            task = 'T1toT2'
        # if name != 'HisGAN_Baseline':
        #     continue
        print(name, task)
        mses = []
        psnrs = []
        ssims = []

        i = 80
        real_A_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_real_A.npy'.format(i, slc)
        real_B_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_real_B.npy'.format(i, slc)
        fake_B_p = idx + '/npy/' + 'FeTS21_Validation_{:03d}_{}.npy_fake_B.npy'.format(i, slc)
        real_A = np.load(real_A_p)
        real_B = np.load(real_B_p)
        fake_B = np.load(fake_B_p)
        res = fake_B - real_B

        # real_A = (real_A+1)/2.0
        # real_B = (real_B+1)/2.0
        # fake_B = (fake_B+1)/2.0
        plt.axis('off')
        plt.xticks([])
        plt.imshow(real_A, cmap='gray')
        plt.savefig('./visualres/{}_real_A_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)
        plt.imshow(real_B, cmap='gray')
        plt.savefig('./visualres/{}_real_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)
        plt.imshow(fake_B, cmap='gray')
        plt.savefig('./visualres/{}_fake_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)

        res = fake_B-real_B
        plt.imshow(res, cmap='jet', vmin=-1.0, vmax=1.0)
        # plt.colorbar()
        plt.savefig('./visualres/{}_res_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)

        # assert False


def figureMRtoCT():
    alls = glob.glob('./output/*')
    slc = 77 # slice number
    dataset = list(range(1, 126))
    rem = [67, 68, 69, 72, 74, 76, 77, 83, 85, 89, 91, 92, 99, 103] # The validation set has 111 Images.
    for i in rem:
        dataset.remove(i)

    # MSE
    MSE = torch.nn.MSELoss()

    # SSIM
    from skimage.metrics import structural_similarity as SSIM


    for idx in alls:
        file = idx.split('/')[-1]
        if 'MR' not in file:
            continue
        task = 'MRtoCT'
        name = file[:-9]
        # if name != 'HisGAN_Baseline':
        #     continue
        print(name, task)
        mses = []
        psnrs = []
        ssims = []

        i = '27_7'
        real_A_p = idx + '/npy/' + '{}_real_A.npy'.format(i)
        real_B_p = idx + '/npy/' + '{}_real_B.npy'.format(i)
        fake_B_p = idx + '/npy/' + '{}_fake_B.npy'.format(i)
        real_A = np.load(real_A_p)
        real_B = np.load(real_B_p)
        fake_B = np.load(fake_B_p)
        res = fake_B - real_B

        # real_A = (real_A+1)/2.0
        # real_B = (real_B+1)/2.0
        # fake_B = (fake_B+1)/2.0
        plt.axis('off')
        plt.xticks([])
        plt.imshow(real_A, cmap='gray')
        plt.savefig('./visualres/{}_real_A_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)
        plt.imshow(real_B, cmap='gray')
        plt.savefig('./visualres/{}_real_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)
        plt.imshow(fake_B, cmap='gray')
        plt.savefig('./visualres/{}_fake_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)

        res = fake_B-real_B
        plt.imshow(res, cmap='jet', vmin=-1.0, vmax=1.0)
        # plt.colorbar()
        plt.savefig('./visualres/{}_res_B_{}.png'.format(task, name), bbox_inches='tight', pad_inches=-0.1)

        # assert False




if __name__ == '__main__':
    import fire
    fire.Fire()
