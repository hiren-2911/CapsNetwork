import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def Violin_plot():
    path="/home/dllab-1/Hiren/Code/CycleCNN/csvs/"
    sr_psnr=pd.read_csv(path+"SR_PSNR.csv")
    sr_ssim=pd.read_csv(path+"SR_SSIM.csv")

    bicubic_psnr=pd.read_csv(path+"Bicubic_PSNR.csv")
    bicubic_ssim=pd.read_csv(path+"Bicubic_SSIM.csv")

    fig=plt.figure(figsize=(10,8))
    fig,axes=plt.subplot((422),figsize=(10,8))
    sns.violinplot(data=sr_psnr["PSNR"], split=True, ax=axes[0,0])
    axes[0,0].set_title("CYCLE MSE PSNR")

    ax = fig.add_subplot(412)
    sns.violinplot(data=bicubic_psnr["PSNR"], split=True, ax=ax)
    ax.set_title("Bicubic PSNR")



    fig.tight_layout()
    plt.show()




if __name__=="__main__":
    Violin_plot()