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

        if opt.is_train:
            self.input_dir = opt.input_dir_train #
            self.target_dir = opt.target_dir_train #

            self.target_path_list = sorted(glob(os.path.join(self.target_dir, '*.', + 'npy')))
            self.image_transform = vt.Compose([
                vt.RandomCrop((self.opt.input_resolution, self.opt.input_resolution)), # 
            ])
            print(len(self.target_path_list))

        else:
            self.input_dir = opt.input_dir_test # 
            self.label_dir = opt.target_dir_test #

    def __getitem__(self, index):
        if self.opt.is_train:


