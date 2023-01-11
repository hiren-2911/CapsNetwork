import torch
from Dataset import Images
from utils import save_checkpoint,load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Generator_model import UpGenerator,GRL
from Discriminator import Discriminator
from datetime import datetime
import os
import sys
from patch_dataset import TrainDataset, EvalDataset
from progan_modules import PerceptualLoss
from metrics import psnr,ssim
from torch.optim.lr_scheduler import StepLR

def mseloss(generated,ground):
    loss_mse = (ground - generated) ** 2
    return loss_mse.mean()


def train_fn(Upmodel,Downmodel,grl,Train_loader,Val_loader,opt_D,opt_G,mse,cross_ent,log_folder,log_file_name,epoch):

    Upmodel.train()
    Downmodel.train()
    grl.train()
    print(Upmodel.training)
    print(Downmodel.training)
    loop1 = tqdm(Train_loader, leave=True)
    batch_loss = []
    for idx,data in enumerate(loop1):
        LR, HR = data
        LR = LR.to(config.DEVICE)
        HR = HR.to(config.DEVICE)

        grl_op=grl(LR)
        fake_HR = Upmodel(LR)+grl_op

        D_real = Downmodel(HR)
        D_fake = Downmodel(fake_HR.detach())

        D_real_loss = mse(D_real, torch.ones_like(D_real))
        D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
        D_loss = D_real_loss + D_fake_loss

        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()

        D_fake = Downmodel(fake_HR)

        l2_loss = mse(fake_HR, HR)

        adversarial_loss = 1e-3 * cross_ent(D_fake, torch.ones_like(D_fake))


        g_loss = adversarial_loss + l2_loss

        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()


        #
        # idt_fwd=0#mse(up_forward,HR).mean()
        # idt_bwd=0#mse(down_backward,LR).mean()
        #
        # identity_loss=0#(idt_bwd+idt_fwd)/2

        #perc_loss=perceptualLoss(up_forward,HR)

#1.489


        # if idx % 50 == 0:
        #     img_path = f"{log_folder}/sample/{epoch}/"
        #     if not os.path.exists(img_path):
        #         os.mkdir(img_path)
        #     save_image(up_forward, img_path + f'{idx}' + '_SR.png', nrow=4, normalize=True, range=(-1, 1))
        #     save_image(HR, img_path + f'{idx}' + '_HR.png', nrow=4, normalize=True, range=(-1, 1))

        if idx % 500 == 0:
            checkpoint_path = f"{log_folder}/checkpoint/{epoch}/"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            try:
                upcheckpoint={'start_epoch': epoch,
                            'state_dict': Upmodel.state_dict(),
                            'optimizer_state_dict': opt_G.state_dict()}
                downcheckpoint={'start_epoch': epoch,
                            'state_dict': Downmodel.state_dict(),
                            'optimizer_state_dict': opt_D.state_dict()}
                grlcheckpoint={'start_epoch': epoch,
                            'state_dict': grl.state_dict(),
                            'optimizer_state_dict': opt_G.state_dict()}
                torch.save(upcheckpoint, checkpoint_path + f'{idx}' + '_g.pt')
                torch.save(downcheckpoint, checkpoint_path + f'{idx}' + '_d.pt')
                torch.save(grlcheckpoint, checkpoint_path + f'{idx}' + '_grl.pt')
            except:
                pass

            # sr = up_forward.detach().cpu().permute(0, 2, 3, 1).numpy()
            # hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
            #psnr_score = psnr(sr, hr)/config.BATCH_SIZE
            #state_msg = (f'epoch:{epoch}; iteration {idx}; Loss: {"%.5f" % loss.item()}; psnr:{"%.5f" % psnr_score}')

            log_file = open(log_file_name, "+a")
            new_line=f'adv:{"%.3f"%adversarial_loss.item()};l2:{"%.3f"%l2_loss.item()};g_:{"%.3f"%g_loss.item()}'+"\n"
            # ,perc loss:{"%.3f" % perc_loss.item()}' #,fwd_loss:{"%.3f"%cyc_fwd.item()},bwd_loss:{"%.3f"%cyc_bwd.item()} '
            log_file.write(new_line)
            # new_line = f'Cycle loss:{"%.3f" % cycle_loss.item()},perc loss:{"%.3f" % perc_loss.item()}'+'\n'
            # log_file.write(new_line)
            log_file.close()
            #print(state_msg)
        batch_loss.append(g_loss.item())
    Training_loss=sum(batch_loss)/len(batch_loss)
    loss_file = open(log_file_name.split('.')[0] + "_train_loss.txt", "+a")
    new_line = f'Training Loss :{"%.3f" % Training_loss}' + '\n'
    loss_file.write(new_line)
    loss_file.close()
    print(f'Training Loss: {Training_loss}')

    # Upmodel.eval()
    # Downmodel.eval()
    # grl.eval()
    # print(Upmodel.training)
    # print(Downmodel.training)
    # loop2 = tqdm(Val_loader, leave=True)
    # Val_loss = 0.0
    # Vbatch_loss =[]
    # for idx, data in enumerate(loop2):
    #     LR, HR = data
    #     LR = LR.to(config.DEVICE)
    #     HR = HR.to(config.DEVICE)
    #
    #     grl_op = grl(LR)
    #     fake_HR = Upmodel(LR) + grl_op
    #
    #     D_real = Downmodel(HR)
    #     D_fake = Downmodel(fake_HR.detach())
    #
    #     D_real_loss = mse(D_real, torch.ones_like(D_real))
    #     D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
    #     D_loss = D_real_loss + D_fake_loss
    #
    #     D_fake = Downmodel(fake_HR)
    #
    #     l2_loss = mse(fake_HR, HR)
    #
    #     adversarial_loss = 1e-3 * cross_ent(D_fake, torch.ones_like(D_fake))
    #
    #     loss = adversarial_loss + l2_loss
    #
    #     #loss = (mse_loss) / 0.005 + perc_loss / 1.54
    #
    #     # if idx % 50 == 0:
    #     #     img_path = f"{log_folder}/sample_val/{epoch}/"
    #     #     if not os.path.exists(img_path):
    #     #         os.mkdir(img_path)
    #     #     save_image(up_forward, img_path + f'{idx}' + '_SR.png', nrow=4, normalize=True, range=(-1, 1))
    #     #     save_image(HR, img_path + f'{idx}' + '_HR.png', nrow=4, normalize=True, range=(-1, 1))
    #
    #
    #     Vbatch_loss.append(loss.item())
    #     # sr = up_forward.detach().cpu().permute(0, 2, 3, 1).numpy()
    #     # hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
    #     # psnr_score = psnr(sr, hr)/config.BATCH_SIZE
    #
    # Val_loss=(sum(Vbatch_loss)/len(Vbatch_loss))
    #
    # state_msg=(f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}')
    # print(state_msg)
    #
    # log_file = open(log_file_name+"_train_val_loss", "+a")
    # new_line = f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}'+'\n'
    # log_file.write(new_line)
    # log_file.close()




def main():
    Upmodel=UpGenerator().to(config.DEVICE)
    grl=GRL().to(config.DEVICE)
    Downmodel=Discriminator().to(config.DEVICE)
    opt_G = optim.Adam(list(grl.parameters())+list(Upmodel.parameters()), lr=config.LEARNING_RATE,
                     betas=(0.9, 0.999), weight_decay=1e-4)
    opt_D=optim.Adam(Downmodel.parameters(),lr=config.LEARNING_RATE, betas=(0.9,0.999), weight_decay=1e-4)
    #opt=optim.Adam(list(Downmodel.parameters())+list(Upmodel.parameters()),lr=config.LEARNING_RATE,betas=(0.9, 0.999),weight_decay=1e-3)

    #scheduler = StepLR(opt, step_size=5, gamma=0.99)

    mse=nn.MSELoss()
    cross_ent = nn.BCELoss()
    train_dataset = TrainDataset(config.TRAIN_FILE, patch_size=config.patch_size, scale=config.scale)
    Train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=config.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=config.NUM_WORKERS,)
    eval_dataset = EvalDataset(config.VAL_FILE)
    Val_loader = DataLoader(dataset=eval_dataset, batch_size=4)#config.BATCH_SIZE)
    # Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=config.transforms)
    # val_dataset=Images(LR_root_dir=config.VAL_DIR+"/LR",HR_root_dir=config.VAL_DIR+"/HR",transfroms=config.transforms)
    # Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    # Val_loader=DataLoader(dataset=val_dataset,batch_size=config.BATCH_SIZE,pin_memory=True,shuffle=False)
    #perceptualLoss = PerceptualLoss(requires_grad=True).cuda()
    log_file = open("/home/ml/Hiren/Code/CycleCNN/Logs/log.txt", "w")
    log_file.write("Starting Training Try 1 \n")
    log_file.close()
    date_time = datetime.now()
    trial_name="Patch_ycbcr_Cycle_dense_GAN"
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
        #scheduler.step()
        #print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        train_fn(Upmodel,Downmodel,grl,Train_loader,Val_loader,opt_D,opt_G,mse,cross_ent,log_folder,log_file_name,epoch)
        #state_msg = (f'Val_epoch:{epoch};Val_psnr:{psnr_score}; SSIM:{ssim_score}')
        #print(state_msg)



if __name__=="__main__":
    main()