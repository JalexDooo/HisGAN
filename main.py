from data.brain_dataset import BrainDataset
from config.brain_config import BrainConfig
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import models
import glob
import os

# A: t1, B: t2
# param_group['lr'] = round(INIT_LR * np.power( 1 - (epoch) / MAX_EPOCHES ,power),8)
# newLR = initLR * (1-epoch/max_epoch)^0.9

def train(**kwargs): # cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train
    brain_config = BrainConfig()
    brain_config._parse(kwargs)
    brain_dataset = BrainDataset(brain_config)
    brain_dataset = DataLoader(brain_dataset, batch_size=brain_config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = getattr(models, brain_config.model)(brain_config)
    # model = CycleGANModel(brain_config)
    model.setup(brain_config)

    for epoch in range(brain_config.epoch_count, brain_config.n_epochs+brain_config.n_epochs_decay+1):
        model.update_learning_rate()
        if hasattr(model, 'alpha'):
            model.alpha = [0.0]
            print('Alpha updated.')
        for i, data in enumerate(brain_dataset):
            model.set_input(data)
            model.optimize_parameters()
            # raise Exception('my break')
            if i % 10 ==0:
                print('Epoch: {}, i: {}, loss: {}'.format(epoch, i, model.get_current_losses()))

        # if epoch % 2 == 0:
        print('saving model')
        model.save_networks('iter_{}'.format(epoch))
            
    model.save_networks('latest')

def predict(**kwargs):
    print('kwargs: {}'.format(kwargs))
    brain_config = BrainConfig()
    brain_config._parse(kwargs)
    brain_config.isTrain = False

    brain_dataset = BrainDataset(brain_config)
    brain_dataset = DataLoader(brain_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    model = getattr(models, brain_config.model)(brain_config)
    model.setup(brain_config)
    model.eval()

    if brain_config.task == 'AtoB':
        task = '{}_to_{}'.format(brain_config.A, brain_config.B)
    else:
        task = '{}_to_{}'.format(brain_config.B, brain_config.A)

    output_path = './output/{}_{}'.format(brain_config.model, task)
    image_path = output_path+'/image'
    npy_path = output_path+'/npy'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if not os.path.exists(npy_path):
        os.mkdir(npy_path)


    for i, data in enumerate(brain_dataset):
        i = data['name'][0]
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()

        real_A = visuals['real_A'].permute(0, 2, 3, 1)[0, :, :, 0]
        real_A = real_A.data.detach().numpy()
        np.save(npy_path+'/{}_real_A.npy'.format(i), real_A)
        real_A = (real_A+1)/2.0*255.0
        # real_A = ((real_A-real_A.min())/((real_A.max()-real_A.min())/255))
        image = Image.fromarray(real_A).convert('L')
        image.save(image_path+'/{}_real_A.png'.format(i))
        # plt.imshow(real_A, cmap='gray')
        # plt.show()

        fake_B = visuals['fake_B'].permute(0, 2, 3, 1)[0, :, :, 0]
        fake_B = fake_B.data.detach().numpy()
        np.save(npy_path+'/{}_fake_B.npy'.format(i), fake_B)
        fake_B = (fake_B+1)/2.0*255.0
        # fake_B = ((fake_B-fake_B.min())/((fake_B.max()-fake_B.min())/255))
        image = Image.fromarray(fake_B).convert('L')
        image.save(image_path+'/{}_fake_B.png'.format(i))
        # plt.imshow(fake_B, cmap='gray')
        # plt.show()

        real_B = visuals['real_B'].permute(0, 2, 3, 1)[0, :, :, 0]
        real_B = real_B.data.detach().numpy()
        np.save(npy_path+'/{}_real_B.npy'.format(i), real_B)
        real_B = (real_B+1)/2.0*255.0
        # real_B = ((real_B-real_B.min())/((real_B.max()-real_B.min())/255))
        image = Image.fromarray(real_B).convert('L')
        image.save(image_path+'/{}_real_B.png'.format(i))
        # plt.imshow(real_B, cmap='gray')
        # plt.show()


        #####

        # real_A = visuals['random_AB'].permute(0, 2, 3, 1)[0, :, :, 0]
        # real_A = real_A.data.detach().numpy()
        # np.save(npy_path+'/{}_random_AB.npy'.format(i), real_A)
        # real_A = (real_A+1)/2.0*255.0
        # image = Image.fromarray(real_A).convert('L')
        # image.save(image_path+'/{}_random_AB.png'.format(i))


if __name__ == '__main__':
    import fire
    fire.Fire()

"""
# train multi-gpu T1->T2 && Flair->T1ce
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ResCycleGANModel' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ResCycleGANModel' --A='flair' --B='t1ce'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='Pix2pix' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='Pix2pix' --A='flair' --B='t1ce'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MultiCycleGANModel' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MultiCycleGANModel' --A='flair' --B='t1ce'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MedGANModel' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MedGANModel' --A='flair' --B='t1ce'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ARGANModel' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ARGANModel' --A='flair' --B='t1ce'

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_Baseline' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_Baseline' --A='flair' --B='t1ce'

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_Baseline_Histloss' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_Baseline_Histloss' --A='flair' --B='t1ce'

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_EMANet' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_EMANet' --A='flair' --B='t1ce'

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_EMANet_Histloss' --A='t1' --B='t2'
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_EMANet_Histloss' --A='flair' --B='t1ce'

# train multi-gpu MR->CT local host
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ResCycleGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/Brain/' --batch_size=1 --gpu_ids=''
# in server cluster
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ResCycleGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='Pix2pixModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MultiCycleGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MedGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='ARGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='HisGAN_Baseline' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter=200 --n_epochs=20 --n_epochs_decay=20 --continue_train=True

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='MedGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --n_epochs=20 --n_epochs_decay=20
cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='BiCycleGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --n_epochs=20 --n_epochs_decay=20


# predict cpu
python3 main.py predict --gpu_ids='' --model='HisGAN_EMANet_Histloss' --A='t1' --B='t2' --load_iter=200 --dataroot='/Users/jontysun/Downloads/数据集/BrainT1T2FT/npyFTTTest'
python3 main.py predict --gpu_ids='' --model='HisGAN_EMANet_Histloss' --A='flair' --B='t1ce' --load_iter=200 --dataroot='/Users/jontysun/Downloads/数据集/BrainT1T2FT/npyFTTTest'

python3 main.py predict --gpu_ids='' --model='ResCycleGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='Pix2pixModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='MultiCycleGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='MedGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='ARGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='HisGAN_Baseline' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='HisGAN_Baseline_Histloss' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='HisGAN_EMANet' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40
python3 main.py predict --gpu_ids='' --model='HisGAN_EMANet_Histloss' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40

cd /sunjindong/TransforGAN_for_Medical && python3 -u main.py train --model='CycleGANModel' --A='MR' --B='CT' --dataroot='/sunjindong/dataset/Brain' --load_iter='latest' --n_epochs=20 --n_epochs_decay=20 --continue_train=True
python3 main.py predict --gpu_ids='' --model='CycleGANModel' --A='MR' --B='CT' --dataroot='/Users/jontysun/Downloads/数据集/BrainTest' --load_iter=40

"""
