import math
from skimage.metrics import structural_similarity
from PIL import Image
import os
import cv2
import shutil
import sys
import numpy as np
from scipy import signal



def cal_ssim(img1, img2):
    K = [0.01, 0.03]
    L = 255
    kernelX = cv2.getGaussianKernel(11, 1.5)
    window = kernelX * kernelX.T

    M, N = np.shape(img1)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mu1 = signal.convolve2d(img1, window, 'valid')
    mu2 = signal.convolve2d(img2, window, 'valid')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = signal.convolve2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mssim = np.mean(ssim_map)
    return mssim, ssim_map



def single_psnr(img1, img2):
    temp=(np.mean((img1 - img2) ** 2))
    if temp == 0:
        return 100
    else:
        return (20 * math.log10(255.0 / math.sqrt(temp)))

def psnr(img1, img2):
    ps=[]
    for i in range(img1.shape[0]):
        temp=(np.mean((img1 - img2) ** 2))
        if temp == 0:
            ps.append(100)
        else:
            ps.append(20 * math.log10(255.0 / math.sqrt(temp)))
    return sum(ps)


def ssim(im1, im2):
    ss=[]
    #print((im1.shape))
    for i in range(im1.shape[0]):
        grayA = cv2.cvtColor(im1[i, :, :, :], cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(im2[i, :, :, :], cv2.COLOR_BGR2GRAY)
        (score, diff) = structural_similarity(grayA, grayB, full=True)
        ss.append(score)
    return sum(ss)

def single_ssim(im1,im2):

    grayA = cv2.cvtColor(im1[0,:,:,:], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(im2[0,:,:,:], cv2.COLOR_BGR2GRAY)
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    return score


def csv_file():
    img_path="/home/dllab-1/Hiren/Code/CycleCNN/Generated_Results/"
    HR=img_path+"HR/"
    SR=img_path+"SR/"
    #SR="/cluster/home/hirenv/CapsNetwork/Cropped Data/Pranav/DenseNET Outputs/"
    Bicubic=img_path+"Bicubic/"
    lst=os.listdir(HR)
    ss=[]
    ps=[]
    bss=[]
    bps=[]
    for i in lst:
        HR_img=np.array(Image.open(HR+i))
        SR_img=np.array(Image.open(SR+i))
        Bicubic_img = np.array(Image.open(Bicubic + i.split('.')[0]+"_Bicubic.png"))
        ps.append(psnr(HR_img,SR_img))
        ss.append(ssim(HR_img, SR_img))

        bps.append(psnr(HR_img, Bicubic_img))
        bss.append(ssim(HR_img, Bicubic_img))

        np.savetxt("/home/dllab-1/Hiren/Code/CycleCNN/csvs/SR_PSNR.csv",np.asarray(ps),delimiter=',')
        np.savetxt("/home/dllab-1/Hiren/Code/CycleCNN/csvs/SR_SSIM.csv",np.asarray(ss),delimiter=',')
        np.savetxt("/home/dllab-1/Hiren/Code/CycleCNN/csvs//Bicubic_PSNR.csv",np.asarray(bps),delimiter=',')
        np.savetxt("/home/dllab-1/Hiren/Code/CycleCNN/csvs/Bicubic_SSIM.csv",np.asarray(bss),delimiter=',')

    print(float(sum(ps))/float(len(ps)))
    print(float(sum(ss))/float(len(ss)))


def bicubic():
   path="/home/ml/Hiren/Data/Val/HR/"
   dest_path="/home/ml/Hiren/Data/Val/LR/"
   lst=os.listdir(path)
   for i in lst:
       img=cv2.imread(path+i,cv2.IMREAD_COLOR)
       img=cv2.resize(img,(70,70),cv2.INTER_CUBIC)
       cv2.imwrite(dest_path+i,img)

def Brisque_Matrix():
    img_path = "/cluster/home/hirenv/CapsNetwork/Cropped Data/Pranav/DenseNet/"
    HR = img_path + "HR/"
    SR = img_path + "SR/"

# def bicubic():
#    img_path = "/home/dllab-1/Hiren/Code/CycleCNN/Generated_Results/"
#    lst=os.listdir(img_path)
#    lst_lr=[]
#    for i in lst:
#        if(i.split('_')[1].split('.')[0]=="LR"):
#            lst_lr.append(i)
#    for i in lst_lr:
#         img=cv2.imread(img_path+i)
#         img=cv2.resize(img,(img.shape[0]*4,img.shape[1]*4),cv2.INTER_CUBIC)
#         cv2.imwrite(f"/home/dllab-1/Hiren/Code/CycleCNN/Bicubic/{i.split('_')[0]+'_Bicubic.png'}",img)
#def renames():
#    srcpath="/cluster/home/hirenv/CapsNetwork/Cropped Data/Pranav/DenseNet/SR/"
#    destpath="/cluster/home/hirenv/CapsNetwork/Cropped Data/Pranav/DenseNet/SR_/"
#    lst=os.listdir((srcpath))
#   for i in lst:
#        shutil.move(srcpath+i,destpath+i.split(".")[0]+".jpg")



if __name__=="__main__":
    bicubic()