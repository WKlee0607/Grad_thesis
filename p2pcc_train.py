"""
Train code for the Pix2PixCC Model
@author: Hyun-Jin Jeong (https://jeonghyunjin.com, jeong_hj@khu.ac.kr)
Reference:
1) https://github.com/JeongHyunJin/Pix2PixCC
2) https://arxiv.org/pdf/2204.12068.pdf
"""

#==============================================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from p2pcc_options import TrainOption
from p2pcc_options import TestOption

import datetime  
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from p2pcc_net import Discriminator, Generator, Loss
from p2pcc_pipeline import CustomDataset

from p2pcc_utils import Manager, weights_init


### 메모
'''
# train_input : GK2A - random_crop 256 x 256 & norm [-1,1] / train_target : CMAX(KMA) - 275 x 300  / 
1. Train_input shape : (300, 275, 3) - VI, IR, WV
2. Train_target shape : (300, 275) - radar

3. Test_input shape : (720, 576, 3) # -> 300 x 275로 resize만 하면됨. 이미 같은 채널로 묶여 있어서.
4. Test_target shape : (720, 576) # -> 300 x 275로 resize만 하면됨. 이미 같은 채널로 묶여 있어서.

5. IMERGE shape : (2600, 3000)
'''


if __name__ == '__main__':
    
    #--------------------------------------------------------------------------
    start_time = datetime.datetime.now()

    #--------------------------------------------------------------------------
    # Init Conditions Setup
    
    torch.backends.cudnn.benchmark = False
    opt = TrainOption().parse()
    test_opt = TestOption().parse()
    
    device = torch.device('cuda:0')
    dtype = torch.float16 if opt.data_type == 16 else torch.float32

    lr = opt.lr
    batch_sz = opt.batch_size
    manager = Manager(opt)

    # --- Dataset load ---
    dataset = CustomDataset(opt)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_sz,
                             num_workers=opt.n_workers,
                             shuffle=not opt.no_shuffle)
    
    test_dataset = CustomDataset(test_opt)
    test_data_loader = DataLoader(dataset=test_dataset,
                                    batch_size=test_opt.batch_size,
                                    num_workers=test_opt.n_workers,
                                    shuffle=not test_opt.no_shuffle)

    
    
    # --- Network and Optimizer update ---
    G = torch.nn.DataParallel(Generator(opt)).apply(weights_init).to(device=device, dtype=dtype)
    D = torch.nn.DataParallel(Discriminator(opt)).apply(weights_init).to(device=device, dtype=dtype)

    G_optim = torch.optim.Adam(G.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    D_optim = torch.optim.Adam(D.parameters(), lr=lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    criterion = Loss(opt)
    
    
    # --- Resume check ---
    if opt.resume:
        G_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_G.pt'
        D_init_path = opt.model_dir + '/' + str(opt.latest_iter) + '_D.pt'

        if os.path.isfile(G_init_path) and os.path.isfile(D_init_path) :
            init_iter = opt.latest_iter
            print("Resume at iteration: ", init_iter)
            
            G.load_state_dict(torch.load(G_init_path))
            D.load_state_dict(torch.load(D_init_path))

            init_epoch = int(float(init_iter)/(batch_sz*len(data_loader)))
            current_step = int(init_iter)

    else:
        init_epoch = 1
        current_step = 0


    
    
    
    #--------------------------------------------------------------------------
    # Training
    
    total_step = opt.n_epochs * len(data_loader) * batch_sz

    for epoch in range(init_epoch, opt.n_epochs + 1):
        G.train()
        for input, target in tqdm(data_loader):
            current_step += batch_sz
            input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
            
            D_loss, G_loss, target_tensor, generated_tensor = criterion(D, G, input, target)

            G_optim.zero_grad()
            G_loss.backward()
            G_optim.step()

            D_optim.zero_grad()
            D_loss.backward()
            D_optim.step()

            package = {'Epoch': epoch,
                       'current_step': current_step,
                       'total_step': total_step,
                       'D_loss': D_loss.detach().item(),
                       'G_loss': G_loss.detach().item(),
                       'D_state_dict': D.module.state_dict(),
                       'G_state_dict': G.module.state_dict(),
                       'D_optim_state_dict': D_optim.state_dict(),
                       'G_optim_state_dict': G_optim.state_dict(),
                       'target_tensor': target_tensor,
                       'generated_tensor': generated_tensor.detach()
                       }

            manager(package)


    #--------------------------------------------------------------------------
    # Test
        G.eval()
        with torch.no_grad():   
            test_image_dir = os.path.join(test_opt.image_dir, str(epoch))
            os.makedirs(test_image_dir, exist_ok=True)
            test_model_dir = test_opt.model_dir
            
            for input, target, name in tqdm(test_data_loader):
                input, target = input.to(device=device, dtype=dtype), target.to(device, dtype=dtype)
                fake = G(input)
                
                np_fake = fake.cpu().numpy().squeeze()
                np_real = target.cpu().numpy().squeeze()
                
                sav_fake = np.clip(np_fake*np.float(1), -1, 1) * 255
                sav_real = np.clip(np_real*np.float(1), -1, 1) * 255
                    
                manager.save_image(sav_fake, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_fake.png'))
                manager.save_image(sav_real, path=os.path.join(test_image_dir, 'Check_{:d}_'.format(current_step)+ name[0] + '_real.png'))


    #--------------------------------------------------------------------------    
    
    end_time = datetime.datetime.now()
    
    print("Total time taken: ", end_time - start_time)

