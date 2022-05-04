from data.utils import load_path
from data.base_dataset import get_transform, BaseDataset, normalize, standard, transform_resize
from PIL import Image
import os
import torch as t
import numpy as np

class BrainDataset(BaseDataset):
    def __init__(self, conf):
        BaseDataset.__init__(self, conf)
        self.dir_A = os.path.join(conf.dataroot, conf.A)
        self.dir_B = os.path.join(conf.dataroot, conf.B)

        self.A_paths = load_path(self.dir_A)
        self.B_paths = load_path(self.dir_B)

        self.len_A = len(self.A_paths)
        self.len_B = len(self.B_paths)

        self.transform_A = get_transform(self.conf, grayscale=False)
        self.transform_B = get_transform(self.conf, grayscale=False)

        self.transform_resize = transform_resize(self.conf)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        A_path = self.A_paths[(idx%self.len_A)]

        name = A_path.split('/')[-1]
        B_path = os.path.join(self.dir_B, name)

        if self.conf.B == 'CT': # not T1->T2.., is MR->CT, the file naming method is different.
            A_path = self.A_paths[(idx%self.len_A)]
            B_path = self.B_paths[(idx%self.len_B)]
            name = A_path.split('/')[-1][:-4]

        A_img = np.load(A_path)
        B_img = np.load(B_path)

        ######
        '''
        A = np.zeros((self.conf.load_size, self.conf.load_size))
        B = np.zeros((self.conf.load_size, self.conf.load_size))
        A[:A_img.shape[0], :A_img.shape[1]] = A_img
        B[:B_img.shape[0], :B_img.shape[1]] = B_img

        A_img = t.from_numpy(A).float()
        B_img = t.from_numpy(B).float()
        A = (A_img-A_img.mean())/A_img.std()
        B = (B_img-B_img.mean())/B_img.std()
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        '''

        A_img = A_img/A_img.max()*255
        B_img = B_img/B_img.max()*255
        A_img = Image.fromarray(np.uint8(A_img)).convert('L')
        B_img = Image.fromarray(np.uint8(B_img)).convert('L')

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)


        # if A.shape[0] == 1:
        #     A = A.repeat(3, 1, 1)
        # if B.shape[0] == 1:
        #     B = B.repeat(3, 1, 1)
        # return A_path, B_path

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'name': name}

