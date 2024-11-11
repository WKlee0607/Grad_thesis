import os
from os.path import split, splitext
import numpy as np
from glob import glob
from random import randint

import torch
from torch.utils.data import Dataset
from torchvision import transforms as vt

from PIL import Image
#from astropy.io import fits
from scipy.ndimage import rotate


class CustomDataset(Dataset):
    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt

        self.input_dir = opt.input_dir_train #
        self.target_dir = opt.target_dir_train #

        self.input_path_list = sorted(glob(os.path.join(self.input_dir, '*.', 'npy')))
        self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.', + 'npy')))
        print(len(self.target_path_list))

        if opt.is_train:
            self.image_transform = vt.Compose([
                vt.RandomCrop((self.opt.input_resolution, self.opt.input_resolution)), #
                vt.ToTensor(), 
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # mean, var
            ])
        else:
            self.image_transform = vt.Compose([
                vt.RandomCrop((self.opt.input_resolution, self.opt.input_resolution)), #
                vt.ToTensor(), 
                vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # mean, var
            ])

    def __getitem__(self, index):
        IMG_A0 = np.load(self.input_path_list[index], allow_pickle=True).transpose(2, 0, 1)
        IMG_A1 = np.load(self.target_path_list[index], allow_pickle=True).transpose(2, 0, 1)

        input = self.image_transform(IMG_A0)
        target = self.image_transform(IMG_A1)
        
        if self.opt.is_train:
            return input, target
        
        return input, target, splitext(split(self.target_path_list[index])[-1])[0]
        
        
        
    


