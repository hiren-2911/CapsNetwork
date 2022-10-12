import os
from metrics import cal_ssim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
def maps():
    path = "/home/dllab-1/Hiren/Code/CycleCNN/Generated_Results/"
    Hr_path = path + "HR/"
    Bicubic_path = path + "Bicubic/"
    cycle_path = path+"SR/"
    lst = os.listdir(Hr_path)
    # bar = fig.add_axes([0.1, 0.2, 0.3, 0.4])  # ,0.5,0.6,0.7,0.8,0.9,1.0])
    for idx, img in enumerate(lst):
        CNN_img = cv2.imread(path + "SR/" + img, 0)
        hr_img = cv2.imread(Hr_path + img, 0)
        bicubic_img = cv2.imread(Bicubic_path + img.split('.')[0]+"_Bicubic.png", 0)

        shssim_index,srhssim_map=cal_ssim(CNN_img,hr_img)
        sbssim_index, srbssim_map = cal_ssim(CNN_img, bicubic_img)

        #shssim_index, srhssim_map = cal_ssim(srgan_img, hr_img)
        sbssim_index, bhssim_map = cal_ssim(bicubic_img, hr_img)
        #sbssim_index, srbssim_map = cal_ssim(srgan_img, bicubic_img)

        hr_img = cv2.imread(Hr_path + img, cv2.IMREAD_COLOR)
        bicubic_img = cv2.imread(Bicubic_path + img.split('.')[0]+"_Bicubic.png", cv2.IMREAD_COLOR)
        CYCLE_img = cv2.imread(path + "SR/" + img)

        fig, axes = plt.subplots(2, 3, figsize=(20, 20))
        fig.tight_layout()

        axes[0][0].imshow(cv2.cvtColor(hr_img,cv2.COLOR_BGR2RGB))
        axes[0][0].set_title("HR Image")
        axes[0][1].imshow(cv2.cvtColor(CYCLE_img,cv2.COLOR_BGR2RGB))
        axes[0][1].set_title("CYCLECNN Image")
        axes[0][2].imshow(cv2.cvtColor(bicubic_img,cv2.COLOR_BGR2RGB))
        axes[0][2].set_title("Bicubic Image")



        axes[1][0].imshow(srhssim_map)
        axes[1][0].set_title("HR-SR")

        axes[1][1].imshow(bhssim_map)
        axes[1][1].set_title("HR-Bicubic")

        axes[1][2].imshow(srbssim_map)
        axes[1][2].set_title("SR-Bicubic")

        plt.savefig(f"/home/dllab-1/Hiren/Code/CycleCNN/Generated_Results/SSIM_MAPS/{img}")

        plt.close()
        if(idx%50==0):
            print(f"{idx}/1000 Done")


if __name__=="__main__":
    maps()