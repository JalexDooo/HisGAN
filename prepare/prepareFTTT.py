import glob
from os import pread
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

def load_nii_to_array(path):
    image = nib.load(path)
    image = image.get_fdata()
    image = np.array(image)
    return image

def nii2npy(output, idx=82): # 72, 77, 82
    path ='/Users/jontysun/Downloads/数据集/BrainT1T2FT/MICCAI_FeTS2021_TrainingData'
    # path ='/Users/jontysun/Downloads/数据集/BrainT1T2FT/MICCAI_FeTS2021_ValidationData'
    paths = glob.glob(path+'/*')
    for p in paths:
        ps = glob.glob(p+'/*.nii.gz')
        name = p.split('/')[-1]
        print(name)
        for i in ps:
            image3d = load_nii_to_array(i)
            if 't1.nii' in i:
                tab = 't1'
            elif 't2.nii' in i:
                tab = 't2'
            elif 't1ce.nii' in i:
                tab = 't1ce'
            elif 'flair.nii' in i:
                tab = 'flair'
            else: # seg
                continue 
            image2d = image3d[:, :, idx]
            # plt.imshow(image2d)
            # plt.show()
            np.save('{}/{}/{}_{}'.format(output, tab, name, idx), image2d)
        # break
        

def test(path):
    image = np.load(path)
    image = Image.fromarray(image)
    image.show()



if __name__ == '__main__':
    nii2npy('/Users/jontysun/Downloads/数据集/BrainT1T2FT/npyFTTT') # train set
    # nii2npy('/Users/jontysun/Downloads/数据集/BrainT1T2FT/npyFTTTest') # test set
