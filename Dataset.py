import torchvision.transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np
import config
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from imagproc import bgr_to_ycbcr,image_to_tensor
from utils_srdense import convert_ycbcr_to_rgb, preprocess,convert_rgb_to_y

import sys
import PIL.Image as pil_image


class Images(Dataset):
    def __init__(self,LR_root_dir ,HR_root_dir,transfroms=None):
        self.LR_root_dir=LR_root_dir
        self.HR_root_dir=HR_root_dir
        self.transforms=transfroms

        self.LR_images=os.listdir(self.LR_root_dir)
        self.HR_images=os.listdir(self.HR_root_dir)
        self.length=len(self.HR_images)
    def __len__(self):
        return self.length


    def __getitem__(self, index):
        LR_img=self.LR_images[index]
        HR_img=self.HR_images[index]
        name=HR_img
        LR_img_path=os.path.join(self.LR_root_dir,LR_img)
        LR_img=np.array(Image.open(LR_img_path))#.convert("RGB"))
        HR_img_path=os.path.join(self.HR_root_dir,HR_img)
        HR_img=np.array(Image.open(HR_img_path))#.convert("RGB"))
        HR_orig=HR_img
        LR_orig=LR_img
        LR_img = convert_rgb_to_y(LR_img)/255.0
        HR_img = convert_rgb_to_y(HR_img)/255.0

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        # LR = image_to_tensor(LR_img, False, False)
        # HR = image_to_tensor(HR_img, False, False)



        if self.transforms:
            augmentations=self.transforms(image=LR_img,image1=HR_img,image2=HR_orig)
            LR=augmentations["image"].float()
            HR=augmentations["image1"].float()
        #     #HR_orig=augmentations["image2"]
        # #     #LR_img=LR_img.double()
        # #     #HR_img=HR_img.double()
        return LR,HR

def Mean_std():
    Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=None)#config.transforms)
    Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    for LR,HR in tqdm(Train_loader):
        HR=HR/255.0
        psum += HR.sum(axis=[0, 2, 3])
        psum_sq += (HR ** 2).sum(axis=[0, 2, 3])

    count = 10000 * 280 * 280

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))

    print(psum)
    print(psum_sq)




IMG_MEAN = [0.6448, 0.3841, 0.2019]
IMG_STD = [0.1238, 0.1057, 0.0832]

def denormalize(x, mean=IMG_MEAN, std=IMG_STD):
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)




def test():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=config.transforms)
    Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    dataiter=iter(Train_loader)
    LR,HR,hr_orig,name=dataiter.next()
    print(hr_orig.shape)
    hr, ycbcr = preprocess(hr_orig[0,:,:,:], device)

    preds = HR[0,:,:,:].mul(255.0).cpu().numpy().squeeze(0)#.squeeze(0)
    print(preds.shape)
    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    print(output.shape)
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)

    output = pil_image.fromarray(output)

    output.save(f"/home/ml/Hiren/Code/{name[0]}")





    # torchvision.utils.save_image(LR, "/home/ml/Hiren/Code/temp_LR.jpg")#, normalize=True, range=(-1, 1))
    # torchvision.utils.save_image(HR,"/home/ml/Hiren/Code/temp_HR.jpg")#, normalize=True,range=(-1,1))




if __name__=="__main__":
    test()


