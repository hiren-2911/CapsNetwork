import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
from Generator_model import UpGenerator, DownGenerator
import numpy as np
from PIL import Image
from Dataset import Images
import config
from torchvision.utils import save_image
from metrics import psnr

import utils
import os



def test_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Images(LR_root_dir=config.VAL_DIR + "/LR", HR_root_dir=config.VAL_DIR + "/HR")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    ##dataset = testOnly_data(LR_path = "/cluster/home/hirenv/CapsNetwork/Data/TEST/LR/", in_memory = False, transform = None)
    # oader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)

    Upmodel=UpGenerator().to(config.DEVICE)
    Upmodel.load_state_dict(torch.load('/home/dllab-1/Hiren/Code/CycleCNN/trial_Test1_2022-09-25_5_29/checkpoint/12/600_g.pt'))
    generator = Upmodel.to(device)
    generator.eval()
    img_path="/home/dllab-1/Hiren/Code/CycleCNN/Generated_Results/"
    psnr_lst=[]

    with torch.no_grad():
        for i, (LR,HR) in enumerate(loader):
            lr = LR.to(device)
            hr = HR.to(device)

            output = generator(lr)
            psnr_lst.append(psnr(output.cpu().numpy(),hr.cpu().numpy()))
            save_image(output, img_path+ "SR/"+ f'{i}' +'.png', normalize=True, range=(-1, 1))
            save_image(hr, img_path+'HR/' + f'{i}' + '.png', normalize=True, range=(-1, 1))
            save_image(lr, img_path+ "LR/"+ f'{i}' + '.png', normalize=True, range=(-1, 1))

    print(sum(psnr_lst)/len(psnr_lst))




if __name__=="__main__":
    test_only()