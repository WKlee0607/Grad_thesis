from p2pcc_options import TestOption
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
from glob import glob
from tqdm import tqdm

from PIL import Image

import torch
from torch.utils.data import DataLoader

from p2pcc_pipeline import CustomDataset
from p2pcc_net import Generator
from p2pcc_utils import Manager

if __name__ == '__main__':
    opt = TestOption().parse()
    #--------------------------------------------------------------------------
    
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    
    
    STD = opt.dataset_name
    EPOCH = opt.test_epoch

    dataset = CustomDataset(opt)
    test_data_loader = DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    path_model = './checkpoints/{}/Model/{}_G.pt'.format(str(STD), str(EPOCH))
    dir_image_save = './checkpoints/{}/Image/Test/{}'.format(str(STD), str(EPOCH))
    if os.path.isdir(dir_image_save) == True:
        pass
    else:                                             
        os.makedirs(dir_image_save, exist_ok=True)
        
    G = torch.nn.DataParallel(Generator(opt)).to(device)
    G.load_state_dict(torch.load(path_model))

    G.eval()
    manager = Manager(opt)
    for input, target, name in tqdm(test_data_loader):
        input, target = input.to(device), target.to(device)

        fake = G(input)

        np_fake = fake.cpu().numpy().squeeze()
        np_real = target.cpu().numpy().squeeze()

        np_fake = fake.cpu().numpy().squeeze()
        np_fake = np.clip(np_fake, -1, 1)

        if len(np_fake.shape) == 3:
            np_fake = np_fake.transpose(1, 2 ,0)        
                    
        np_fake = np.asarray(np_fake, np.uint8)
        pil_image = Image.fromarray(np_fake)
        pil_image.save(os.path.join(dir_image_save, name[0] + '_AI.jpg'))

        
        
