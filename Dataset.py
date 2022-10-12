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
        LR_img_path=os.path.join(self.LR_root_dir,LR_img)
        LR_img=np.array(Image.open(LR_img_path).convert("RGB"))
        HR_img_path=os.path.join(self.HR_root_dir,HR_img)
        HR_img=np.array(Image.open(HR_img_path).convert("RGB"))



        if self.transforms:
            augmentations=self.transforms(image=LR_img,image1=HR_img)
            LR_img=augmentations["image"]
            HR_img=augmentations["image1"]
            #LR_img=LR_img.double()
            #HR_img=HR_img.double()
        return LR_img,HR_img

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
def test():
    Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=config.transforms)
    Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    dataiter=iter(Train_loader)
    LR,HR=dataiter.next()
    #print(LR.size())
    torchvision.utils.save_image(LR, "/home/ml/Hiren/Code/temp_LR.jpg", normalize=True, range=(-1, 1))
    torchvision.utils.save_image(HR,"/home/ml/Hiren/Code/temp_HR.jpg", normalize=True,range=(-1,1))




if __name__=="__main__":
    test()


