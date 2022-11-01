import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
# from Generator_model import UpGenerator, DownGenerator, GRL
from UGEN import UNET
import numpy as np
from PIL import Image
from Dataset import Images,denormalize
import config
from torchvision.utils import save_image
from metrics import psnr
import PIL.Image as pil_image
import utils
import os
from utils_srdense import convert_ycbcr_to_rgb, preprocess,calc_psnr



def test_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Images(LR_root_dir=config.VAL_DIR + "/LR", HR_root_dir=config.VAL_DIR + "/HR",transfroms=config.transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    ##dataset = testOnly_data(LR_path = "/cluster/home/hirenv/CapsNetwork/Data/TEST/LR/", in_memory = False, transform = None)
    # oader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)

    model=UNET(in_channels=1, out_channels=1).to(config.DEVICE)

    checkpoint = torch.load('/home/ml/Hiren/Code/CycleCNN/trial_try1_UNET_GAN_2022-11-01_16_56/checkpoint/5/500_g.pt')
    model.load_state_dict(checkpoint['state_dict'])

    #model.load_state_dict(torch.load())
    #grl.load_state_dict(torch.load('/home/ml/Hiren/Code/CycleCNN/trial_GRL_BLOCK_try1_2022-10-19_15_29/checkpoint/92/2400_grl.pt'))
    #generator = Upmodel.to(device)
    test_LR="/home/ml/Hiren/Data/Test/LR/"
    test_HR = "/home/ml/Hiren/Data/Test/HR/"
    test_bicubic="/home/ml/Hiren/Data/Test/Bicubic/"
    lst=os.listdir(test_LR)


    model.eval()

    img_path="/home/ml/Hiren/Code/CycleCNN/trial_try1_UNET_GAN_2022-11-01_16_56/Generated/"
    psnr_lst=[]

    with torch.no_grad():
        #for idx, (LR,HR,name) in enumerate(loader):
        for idx,image in enumerate(lst):
            #lr = LR.permute((0, 3, 1, 2))
            lr = pil_image.open(test_LR+image).convert('RGB')
            hr= pil_image.open(test_HR+image).convert('RGB')
            bicubic=pil_image.open(test_bicubic+image).convert('RGB')
            hr1=hr
            # hr = image.resize((280, 280), resample=pil_image.BICUBIC)
            # lr = hr.resize((70,70), resample=pil_image.BICUBIC)
            # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            # bicubic.save(args.image_file.replace('test', '_bicubic_{}'.format(args.scale)))
            #bicubic.save(args.bicubic_dir + 'bicubic' + i)
            # lr = LR.to(device)
            #
            # hr = HR.to(device)

            lr, _ = preprocess(lr, device)
            hr, _ = preprocess(hr, device)
            bi, ycbcr = preprocess(bicubic, device)

            output = model(lr).clamp(0.0, 1.0)

            psnr = calc_psnr(hr, output)
            # avgpsnr+=psnr
            psnr_lst.append(psnr)
            print('PSNR: {:.2f}'.format(psnr))

            preds = output.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            #ssim_cal = ssim(output, hr1)
            #avgssim.append(ssim_cal)
            output = pil_image.fromarray(output)
            # output.save(args.outputs_dir+'{}.jpg'.format(j))

            output.save(f"/home/ml/Hiren/Code/CycleCNN/trial_try1_UNET_GAN_2022-11-01_16_56/Generated/{image}")


            # lr=denormalize(lr)
            # output=denormalize(output)
            # hr=denormalize(hr)
            # if(idx%50==0):
            #     print(f'{idx} done')
            # psnr_lst.append(psnr(output.cpu().numpy(),hr.cpu().numpy()))
            # save_image(output, img_path+ "SR/"+f'{name[0]}')#, normalize=True, range=(-1, 1))
            # save_image(hr, img_path +"HR/" +f'{name[0]}')#, normalize=True, range=(-1, 1))
            # save_image(lr, img_path+"LR/" +f'{name[0]}')#, normalize=True, range=(-1, 1))

    print(sum(psnr_lst)/len(psnr_lst))




if __name__=="__main__":
    test_only()