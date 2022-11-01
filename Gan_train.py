import torch
from Dataset import Images,denormalize
from utils import save_checkpoint,load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from Discriminator import Discriminator
from losses_srgan import TVLoss, perceptual_loss
from UGEN import UNET
from datetime import datetime
import os
import sys
from progan_modules import PerceptualLoss
from metrics import psnr,ssim,grp_psnr
from torch.optim.lr_scheduler import StepLR
from vgg19 import vgg19

def train_fn(Upmodel,Downmodel,Train_loader,Val_loader,opt_G,opt_D,mse,cross_ent,VGG_loss,log_folder,log_file_name,epoch):

    Upmodel.train()
    Downmodel.train()
    print(Upmodel.training)
    print(Downmodel.training)
    loop1 = tqdm(Train_loader, leave=True)
    batch_loss = []
    for idx,(LR,HR) in enumerate(loop1):
        LR=LR.to(config.DEVICE)
        HR=HR.to(config.DEVICE)
        # Train Discriminators H and Z
        #with torch.cuda.amp.autocast():
        fake_HR = Upmodel(LR) #upsampled
        D_real = Downmodel(HR) #30x30
        D_fake = Downmodel(fake_HR.detach())  #121x121

        D_real_loss = mse(D_real, torch.ones_like(D_real))
        D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
        D_loss = D_real_loss + D_fake_loss


        opt_D.zero_grad()
        D_loss.backward()
        opt_D.step()
        #d_scaler.scale(D_loss).backward()
        #d_scaler.step(opt_disc)
        #d_scaler.update()

        # Train Generators H and Z
        #with torch.cuda.amp.autocast():
            # adversarial loss for both generators
        D_fake = Downmodel(fake_HR)
        #loss_G= mse(D_fake, torch.ones_like(D_fake))

        l2_loss=mse(fake_HR,HR)
        #_percep_loss, hr_feat, sr_feat = VGG_loss((HR + 1.0) / 2.0, (fake_HR + 1.0) / 2.0, layer='relu5_4')

        #percep_loss =  0.006 * _percep_loss
        adversarial_loss = 1e-3 * cross_ent(D_fake, torch.ones_like(D_fake))
        #total_variance_loss = args.tv_loss_coeff * tv_loss(args.vgg_rescale_coeff * (hr_feat - sr_feat) ** 2)

        g_loss = adversarial_loss + l2_loss # percep_loss

        #G_loss=loss_G+l2_loss


        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()
        #g_scaler.scale(G_loss).backward()
        #g_scaler.step(opt_gen)
        #g_scaler.update()

        # if idx % 50 == 0:
            # img_path = f"{log_folder}/sample/{epoch}/"
            # if not os.path.exists(img_path):
            #     os.mkdir(img_path)



            # save_image(denormalize(fake_HR.detach()), img_path + f'{idx}' + '_SR.png', nrow=4)#  , normalize=True, range=(-1, 1))
            # save_image(denormalize(HR.detach()), img_path + f'{idx}' + '_HR.png', nrow=4)#    , normalize=True, range=(-1, 1))


        if idx % 500 == 0:
            checkpoint_path = f"{log_folder}/checkpoint/{epoch}/"
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)

            gen_state = {'epoch': epoch + 1, 'state_dict': Upmodel.state_dict(),
                     'optimizer': opt_G.state_dict()}
            dict_state = {'epoch': epoch + 1, 'state_dict': Downmodel.state_dict(),
                         'optimizer': opt_D.state_dict()}

            try:
                torch.save(gen_state, checkpoint_path + f'{idx}' + '_g.pt')
                torch.save(dict_state, checkpoint_path + f'{idx}' + '_d.pt')
                #torch.save(grl.state_dict(), checkpoint_path + f'{idx}' + '_grl.pt')
            except:
                pass

            # sr = fake_HR.detach().cpu().permute(0, 2, 3, 1).numpy()
            # hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
            psnr_score=grp_psnr(fake_HR.detach().cpu().numpy(),HR.detach().cpu().numpy())
            #psnr_score = psnr(sr, hr)/config.BATCH_SIZE
            state_msg = (f'Train psnr:{"%.5f" % psnr_score}')
            print(state_msg)

            log_file = open(log_file_name, "+a")
            new_line = f'G:{"%.3f" % g_loss.item()}; D:{"%.3f" % D_loss.item()};l2:{"%.3f" % l2_loss.item()}'+ '\n'#;perc:{"%.3f" % _percep_loss.item()}' + '\n'  # ,perc loss:{"%.3f" % perc_loss.item()}' #,fwd_loss:{"%.3f"%cyc_fwd.item()},bwd_loss:{"%.3f"%cyc_bwd.item()} '
            log_file.write(new_line)
            # # new_line = f'Cycle loss:{"%.3f" % cycle_loss.item()},perc loss:{"%.3f" % perc_loss.item()}'+'\n'
            # # log_file.write(new_line)
            log_file.close()
            # print(state_msg)
    #     batch_loss.append(loss.item())
    # Training_loss = sum(batch_loss) / len(batch_loss)
    # loss_file = open(log_file_name.split('.')[0] + "_train_loss.txt", "+a")
    # new_line = f'Training Loss :{"%.3f" % Training_loss}' + '\n'
    # loss_file.write(new_line)
    # loss_file.close()
    # print(f'Training Loss: {Training_loss}')\
    Upmodel.eval()
    Downmodel.eval()
    print(Upmodel.training)
    print(Downmodel.training)
    loop2 = tqdm(Val_loader, leave=True)

    for idx, (LR, HR) in enumerate(loop2):
        LR = LR.to(config.DEVICE)
        HR = HR.to(config.DEVICE)

        fake_HR = Upmodel(LR)  # upsampled
        D_real = Downmodel(HR)  # 30x30
        D_fake = Downmodel(fake_HR.detach())  # 121x121

        D_real_loss = mse(D_real, torch.ones_like(D_real))
        D_fake_loss = mse(D_fake, torch.zeros_like(D_fake))
        D_loss = D_real_loss + D_fake_loss

        D_fake = Downmodel(fake_HR)
        # loss_G= mse(D_fake, torch.ones_like(D_fake))

        l2_loss = mse(fake_HR, HR)
       # _percep_loss, hr_feat, sr_feat = VGG_loss((HR + 1.0) / 2.0, (fake_HR + 1.0) / 2.0, layer='relu5_4')

        # if idx % 50 == 0:
        #     img_path = f"{log_folder}/sample_val/{epoch}/"
        #     if not os.path.exists(img_path):
        #         os.mkdir(img_path)
        #     save_image(up_forward, img_path + f'{idx}' + '_SR.png', nrow=4, normalize=True, range=(-1, 1))
        #     save_image(HR, img_path + f'{idx}' + '_HR.png', nrow=4, normalize=True, range=(-1, 1))

        # Vbatch_loss.append(loss.item())
        # sr = up_forward.detach().cpu().permute(0, 2, 3, 1).numpy()
        # hr = HR.detach().cpu().permute(0, 2, 3, 1).numpy()
        # psnr_score = psnr(sr, hr) / config.BATCH_SIZE

        psnr_score=grp_psnr(fake_HR.detach().cpu().numpy(),HR.detach().cpu().numpy())
        # psnr_score = psnr(sr, hr)/config.BATCH_SIZE
        state_msg = (f'Val psnr:{"%.5f" % psnr_score}')
        print(state_msg)

    # Val_loss = (sum(Vbatch_loss) / len(Vbatch_loss))
    #
    # state_msg = (f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}; psnr:{"%.3f" % psnr_score}')
    # print(state_msg)
    #
    # log_file = open(log_file_name + "_train_val_loss", "+a")
    # new_line = f'epoch:{epoch}; Training_loss:{Training_loss}; Val_loss:{Val_loss}' + '\n'
    # log_file.write(new_line)
    # log_file.close()



def main():
    Upmodel=UNET(in_channels=1, out_channels=1).to(config.DEVICE)
    Downmodel=Discriminator(in_channels=1).to(config.DEVICE)
    opt_G = optim.Adam(Upmodel.parameters(), lr=config.LEARNING_RATE,
                     betas=(0.9, 0.999), weight_decay=1e-3)

    opt_D = optim.Adam(Downmodel.parameters(), lr=config.LEARNING_RATE,
                       betas=(0.9, 0.999), weight_decay=1e-3)

    #opt=optim.Adam(list(Downmodel.parameters())+list(Upmodel.parameters()),lr=config.LEARNING_RATE,betas=(0.9, 0.999),weight_decay=1e-3)

    #scheduler = StepLR(opt, step_size=5, gamma=0.99)

    mse=nn.MSELoss()
    vgg_net = vgg19().to(config.DEVICE)
    vgg_net = vgg_net.eval()

    VGG_loss = perceptual_loss(vgg_net)
    cross_ent = nn.BCELoss()
    start_epoch=0

    if config.LOAD_MODEL:
        print("Loading Model")
        Upmodel,opt_G,start_epoch=config.load_checkpoint(Upmodel,opt_G,config.GEN)
        Downmodel, opt_D, start_epoch = config.load_checkpoint(Downmodel, opt_D, config.DISC)
        # Upmodel.load_state_dict(torch.load(config.GEN))
        # Downmodel.load_state_dict(torch.load(config.DISC))
        print("Model Loaded Successfully")

    Train_dataset=Images(LR_root_dir=config.TRAIN_DIR+"/LR",HR_root_dir=config.TRAIN_DIR+"/HR",transfroms=config.transforms)
    val_dataset=Images(LR_root_dir=config.VAL_DIR+"/LR",HR_root_dir=config.VAL_DIR+"/HR",transfroms=config.transforms)
    Train_loader=DataLoader(dataset=Train_dataset,batch_size=config.BATCH_SIZE,shuffle=True,pin_memory=True,num_workers=config.NUM_WORKERS)
    Val_loader=DataLoader(dataset=val_dataset,batch_size=config.BATCH_SIZE,pin_memory=True,shuffle=False)
    # perceptualLoss = PerceptualLoss(requires_grad=True).cuda()
    log_file = open("/home/ml/Hiren/Code/CycleCNN/Logs/log.txt", "w")
    log_file.write("Starting Training Try 1 \n")
    log_file.close()
    date_time = datetime.now()
    trial_name="try1_UNET_GAN"
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

    for epoch in range(start_epoch,config.NUM_EPOCHS):
        #scheduler.step()
        #print('Epoch:', epoch, 'LR:', scheduler.get_last_lr())
        train_fn(Upmodel,Downmodel,Train_loader,Val_loader,opt_G,opt_D,mse,cross_ent,VGG_loss,log_folder,log_file_name,epoch)
        #state_msg = (f'Val_epoch:{epoch};Val_psnr:{psnr_score}; SSIM:{ssim_score}')
        #print(state_msg)



if __name__=="__main__":
    main()