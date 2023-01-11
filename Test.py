import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils
from torch.utils.data import DataLoader
from Generator_model import UpGenerator, DownGenerator, GRL
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
from patch_dataset import TrainDataset, EvalDataset
from torchsummary import summary
from utils_srdense import convert_ycbcr_to_rgb, preprocess,calc_psnr,convert_rgb_to_ycbcr



def test_only():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset = Images(LR_root_dir=config.VAL_DIR + "/LR", HR_root_dir=config.VAL_DIR + "/HR",transfroms=config.transforms)
    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    # train_dataset = TrainDataset("/home/ml/Hiren/Data/Train/test.h5", patch_size=config.patch_size, scale=config.scale)
    # loader = DataLoader(dataset=train_dataset,
    #                           batch_size=config.BATCH_SIZE,
    #                           shuffle=True,
    #                           num_workers=config.NUM_WORKERS)

    ##dataset = testOnly_data(LR_path = "/cluster/home/hirenv/CapsNetwork/Data/TEST/LR/", in_memory = False, transform = None)
    # oader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = args.num_workers)

    model = UpGenerator().to(config.DEVICE)
    grl = GRL().to(config.DEVICE)

    checkpoint = torch.load('/home/ml/Hiren/Code/CycleCNN/trial_Patch_ycbcr_Cycle_dense_GAN_2023-01-11_15_16/checkpoint/6/0_g.pt')
    model.load_state_dict(checkpoint['state_dict'])
    checkpoint = torch.load(
        '/home/ml/Hiren/Code/CycleCNN/trial_Patch_ycbcr_Cycle_dense_GAN_2023-01-11_15_16/checkpoint/6/0_grl.pt')
    grl.load_state_dict(checkpoint['state_dict'])
    # print_summary(model)
    # print_summary(grl)
    # idx=0
    # for name, param in model.named_parameters():
    #     print(name)
    #     print(param.shape)
    #     if((idx+1)%5==0):
    #         sys.exit()
    #     idx=idx+1
    #model.load_state_dict(torch.load())
    #grl.load_state_dict(torch.load('/home/ml/Hiren/Code/CycleCNN/trial_GRL_BLOCK_try1_2022-10-19_15_29/checkpoint/92/2400_grl.pt'))
    #generator = Upmodel.to(device)
    test_LR="/home/ml/Hiren/Data/Test/LR/"
    test_HR = "/home/ml/Hiren/Data/Test/HR/"
    test_bicubic="/home/ml/Hiren/Data/Test/Bicubic/"
    lst=os.listdir(test_LR)
    print(len(lst))

    model.eval()

    img_path="/home/ml/Hiren/Code/test/"
    psnr_lst=[]

    with torch.no_grad():
        # for idx, data in enumerate(loader):
         for idx,image in enumerate(lst):
            #lr = LR.permute((0, 3, 1, 2))
            lr = pil_image.open(test_LR + image).convert('RGB')
            hr = pil_image.open(test_HR + image).convert('RGB')
            bicubic = pil_image.open(test_bicubic + image).convert('RGB')
            hr1 = hr
            # hr = image.resize((280, 280), resample=pil_image.BICUBIC)
            # lr = hr.resize((70, 70), resample=pil_image.BICUBIC)
            # bicubic = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
            # bicubic.save(args.image_file.replace('test', '_bicubic_{}'.format(args.scale)))
            # bicubic.save(args.bicubic_dir + 'bicubic' + i)
            # lr = LR.to(device)
            #
            # hr = HR.to(device)

            lr, _ = preprocess(lr, device)
            hr, _ = preprocess(hr, device)
            bi, ycbcr = preprocess(bicubic, device)



            output = (model(lr)+grl(lr)).clamp(0.0, 1.0)

            # grl_op = grl(LR)
            # up_forward = Upmodel(LR) + grl_op
            # down_forward = Downmodel(up_forward)

            # down_backward = Downmodel(HR)
            # up_backward = Upmodel(down_backward)
            #
            # mse_loss = mse(up_forward, HR)
            # cyc_fwd = mse(down_forward, LR)
            # cyc_bwd = mse(up_backward, HR)
            # #
            # cycle_loss = (cyc_bwd + cyc_fwd) / 2



            psnr = calc_psnr(hr, output)
            # avgpsnr+=psnr
            psnr_lst.append(psnr)
            print('PSNR: {:.2f}'.format(psnr))

            preds = output.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
            # print(preds.shape)
            # print(ycbcr[..., 1].shape)
            # sys.exit()
            output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
            output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
            #ssim_cal = ssim(output, hr1)
            #avgssim.append(ssim_cal)
            output = pil_image.fromarray(output)
            # output.save(args.outputs_dir+'{}.jpg'.format(j))

            output.save(f"/home/ml/Hiren/Code/test/{image}")


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

def y_channels():
    src_path="/home/ml/Hiren/Code/Results/RGB/Ycbcr_cycle_dense/"
    dest_path="/home/ml/Hiren/Code/Results/Y_channel/Ycbcr_cycle_dense/"
    lst=os.listdir(src_path)
    for i in lst:
        #print(img)
        img=np.array(Image.open(src_path+i).convert('RGB'))
        y_channel=convert_rgb_to_ycbcr(img)[:,:,0]
        y_channel = pil_image.fromarray(y_channel).convert("L")
        y_channel.save(dest_path+i)


def print_summary(model):
    print("Printing Summary")
    summary(model, input_size=(1, 25, 25))



if __name__=="__main__":
    test_only()