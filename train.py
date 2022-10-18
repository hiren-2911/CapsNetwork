import torch
from Dataset import Images
from utils import save_checkpoint,load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Generator_model import UpGenerator, DownGenerator,GRL
from datetime import datetime
import os
import sys
from progan_modules import PerceptualLoss
from metrics import psnr,ssim
from torch.optim.lr_scheduler import StepLR

def mseloss(generated,ground):
    loss_mse = (ground - generated) ** 2
    return loss_mse.mean()


def train_fn(Upmodel,Downmodel,grl,Train_loader,Val_loader,opt,mse,perceptualLoss,log_folder,log_file_name,epoch):

    Upmodel.train()
    Downmodel.train()
    grl.train()
    print(Upmodel.training)
    print(Downmodel.training)
    loop1 = tqdm(Train_loader, leave=True)
    batch_loss = []
    for idx,(LR,HR) in enumerate(loop1):
        LR=LR.to(config.DEVICE)
        HR=HR.to(config.DEVICE)




        grl_op=grl(LR)
        up_forward = Upmodel(LR)+grl_op
        down_forward=Downmodel(up_forward)

        down_backward=Downmodel(HR)
        up_backward=Upmodel(down_backward)

        mse_loss=mse(up_forward,HR)
        cyc_fwd=mse(down_forward, LR)
        cyc_bwd=mse(up_backward, HR)
        #
        cycle_loss=(cyc_bwd+cyc_fwd)/2
        #
        # idt_fwd=0#mse(up_forward,HR).mean()
        # idt_bwd=0#mse(down_backward,LR).mean()
        #
        # identity_loss=0#(idt_bwd+idt_fwd)/2

        #perc_loss=perceptualLoss(up_forward,HR)

#1.489
        loss=mse_loss+cycle_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if idx % 50 == 0:
            img_path = f"{log_folder}/sample/{epoch}/"
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            save_image(up_forward, img_path + f'{idx}' + '_SR.png', nrow=4, normalize=True, range=(-1, 1))
            save_image(HR, img_path + f'{idx}' + '_HR.png', nrow=4, normalize=True, range=(-1, 1))

        if idx % 100 == 0:
            checkpoint_path = f"{log_folder}/checkpoint/{epoch}/"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            try:
                torch.save(Upmodel.state_dict(), checkpoint_path + f'{idx}' + '_g.pt')
                torch.save(Downmodel.state_dict(), checkpoint_path + f'{idx}' + '_d.pt')
                torch.save(grl.state_dict(), checkpoint_path + f'{idx}' + '_grl.pt')
            except:
                pass

            sr = up_forward.detach().cpu().permute(0, 2, 3, 1).numpy()
            hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
            #psnr_score = psnr(sr, hr)/config.BATCH_SIZE
            #state_msg = (f'epoch:{epoch}; iteration {idx}; Loss: {"%.5f" % loss.item()}; psnr:{"%.5f" % psnr_score}')

            log_file = open(log_file_name, "+a")
            new_line=f'mse:{"%.3f"%mse_loss.item()}; Cycle loss:{"%.3f" % cycle_loss.item()}'# ,perc loss:{"%.3f" % perc_loss.item()}' #,fwd_loss:{"%.3f"%cyc_fwd.item()},bwd_loss:{"%.3f"%cyc_bwd.item()} '
            log_file.write(new_line)
            # new_line = f'Cycle loss:{"%.3f" % cycle_loss.item()},perc loss:{"%.3f" % perc_loss.item()}'+'\n'
            # log_file.write(new_line)
            log_file.close()
            #print(state_msg)
        batch_loss.append(loss.item())
    Training_loss=sum(batch_loss)/len(batch_loss)
    loss_file = open(log_file_name.split('.')[0] + "_train_loss.txt", "+a")
    new_line = f'Training Loss :{"%.3f" % Training_loss}' + '\n'
    loss_file.write(new_line)
    loss_file.close()
    print(f'Training Loss: {Training_loss}')



    Upmodel.eval()
    Downmodel.eval()
    grl.eval()
    print(Upmodel.training)
    print(Downmodel.training)
    loop2 = tqdm(Val_loader, leave=True)
    Val_loss = 0.0
    Vbatch_loss =[]
    for idx,(LR,HR) in enumerate(loop2):
        LR=LR.to(config.DEVICE)
        HR=HR.to(config.DEVICE)

        grl_op = grl(LR)
        up_forward = Upmodel(LR) + grl_op
        down_forward = Downmodel(up_forward)

        down_backward = Downmodel(HR)
        up_backward = Upmodel(down_backward)

        mse_loss = mse(up_forward, HR)
        cyc_fwd = mse(down_forward, LR)
        cyc_bwd = mse(up_backward, HR)

        cycle_loss = (cyc_bwd + cyc_fwd) / 2

        #perc_loss = perceptualLoss(up_forward, HR)
        loss = mse_loss+cycle_loss  #perc=1.33
        #loss = (mse_loss) / 0.005 + perc_loss / 1.54

        if idx % 50 == 0:
            img_path = f"{log_folder}/sample_val/{epoch}/"
            if not os.path.exists(img_path):
                os.mkdir(img_path)
            save_image(up_forward, img_path + f'{idx}' + '_SR.png', nrow=4, normalize=True, range=(-1, 1))
            save_image(HR, img_path + f'{idx}' + '_HR.png', nrow=4, normalize=True, range=(-1, 1))


        Vbatch_loss.append(loss.item())
        sr = up_forward.detach().cpu().permute(0, 2, 3, 1).numpy()
        hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
        psnr_score = psnr(sr, hr)/config.BATCH_SIZE

    Val_loss=(sum(Vbatch_loss)/len(Vbatch_loss))

    state_msg=(f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}; psnr:{"%.3f" % psnr_score}')
    print(state_msg)

    log_file = open(log_file_name+"_train_val_loss", "+a")
    new_line = f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}'+'\n'
    log_file.write(new_line)
    log_file.close()




def main():
    Upmodel=UpGenerator().to(config.DEVICE)
    grl=GRL().to(config.DEVICE)
    Downmodel=DownGenerator().to(config.DEVICE)
    opt = optim.Adam(list(Downmodel.parameters())+list(grl.parameters())+list(Upmodel.parameters()), lr=config.LEARNING_RATE,
                     betas=(0.9, 0.999), weight_decay=1e-3)
    #opt=optim.Adam(list(Downmodel.parameters())+list(Upmodel.parameters()),lr=config.LEARNING_RATE,betas=(0.9, 0.999),weight_decay=1e-3)

    scheduler = StepLR(opt, step_size=5, gamma=0.99)

    mse=nn.L1Loss()

    Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=config.transforms)
    val_dataset=Images(LR_root_dir=config.VAL_DIR+"/LR",HR_root_dir=config.VAL_DIR+"/HR",transfroms=config.transforms)
    Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    Val_loader=DataLoader(dataset=val_dataset,batch_size=8,pin_memory=True,shuffle=False)
    perceptualLoss = PerceptualLoss(requires_grad=True).cuda()
    log_file = open("/home/ml/Hiren/Code/CycleCNN/Logs/log.txt", "w")
    log_file.write("Starting Training Try 1 \n")
    log_file.close()
    date_time = datetime.now()
    trial_name="GRL_BLOCK_try1"
    post_fix = '%s_%s_%d_%d.txt' % (trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d' % (trial_name, date_time.date(), date_time.hour, date_time.minute)

    os.mkdir(log_folder)
#    os.mkdir(log_folder + '/checkpoint')
    os.mkdir(log_folder + '/sample')
    os.mkdir(log_folder + '/sample_val')
    # config_file_name = os.path.join(log_folder, 'train_config_' + post_fix)
    # config_file = open(config_file_name, 'w')
    # config_file.write(str(args))
    # config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_' + post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('Starting Training\n')
    log_file.close()
    loss_file = open(log_file_name.split('.')[0] + "_train_loss.txt", "w")
    loss_file.write('Starting Training\n')
    loss_file.close()

    for epoch in range(config.NUM_EPOCHS):
        scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        train_fn(Upmodel,Downmodel,grl,Train_loader,Val_loader,opt,mse,perceptualLoss,log_folder,log_file_name,epoch)
        #state_msg = (f'Val_epoch:{epoch};Val_psnr:{psnr_score}; SSIM:{ssim_score}')
        #print(state_msg)



if __name__=="__main__":
    main()




