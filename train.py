'''
training for CycleGAN
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from torch.utils.tensorboard import SummaryWriter

from summer_winter_dataset import SummerWinterDataset
from generator import Generator
from discriminator import Discriminator
import config

from tqdm import tqdm
import os
import time
import warnings

import matplotlib.pyplot as plt
from IPython.display import clear_output

def show_real_and_fake(gen_W, gen_S, dataset, clr_output=True):

    if clr_output:
        clear_output(wait=True)
    std = mean = 0.5
    gen_W.eval()
    gen_S.eval()
    plt.figure(figsize=(12, 6))
    for i in range(6):

        real_summ, real_win = dataset[i]
        with torch.no_grad():
            fake_summ = gen_S(torch.unsqueeze(real_win.to(config.DEVICE, dtype=torch.float), 0))
            fake_win = gen_W(torch.unsqueeze(real_summ.to(config.DEVICE, dtype=torch.float), 0))
        fake_summ = fake_summ[0].cpu()
        fake_win = fake_win[0].cpu()


        real_summ = real_summ * std + mean
        fake_summ = fake_summ * std + mean

        real_win = real_win * std + mean
        fake_win = fake_win * std + mean
        
        plt.subplot(4, 6, i+1)
        plt.axis("off")
        plt.imshow(real_summ.permute(1, 2, 0))
        plt.title('Real Summer')

        plt.subplot(4, 6, i+7)
        plt.axis("off")
        plt.imshow(fake_win.permute(1, 2, 0))
        plt.title('Fake winter')

        plt.subplot(4, 6, i+7+6)
        plt.axis("off")
        plt.imshow(real_win.permute(1, 2, 0))
        plt.title('Real Winter')

        plt.subplot(4, 6, i+7+6+6)
        plt.axis("off")
        plt.imshow(fake_summ.permute(1, 2, 0))
        plt.title('Fake Summer')

    plt.show()

def train(disc_W, disc_S,
          gen_W, gen_S,
          optim_disc, optim_gen,
          disc_scaler, gen_scaler,
          dloader,
          writer):

    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    disc_W.train()
    disc_S.train()
    gen_W.train()
    gen_S.train()
    for i, (summer_img, winter_img) in enumerate(tqdm(dloader, leave=False)):
        
        summer_img = summer_img.to(config.DEVICE)
        winter_img = winter_img.to(config.DEVICE)

        # Train Discriminators
        
        
        with torch.cuda.amp.autocast():
            # 1st half of circle
            fake_winter = gen_W(summer_img)
            disc_win_fake = disc_W(fake_winter)
            disc_win_real = disc_W(winter_img)
            
            win_fake_loss = mse(disc_win_fake, torch.zeros_like(disc_win_fake))
            win_real_loss = mse(disc_win_real, torch.ones_like(disc_win_real))
            win_loss = win_fake_loss + win_real_loss
            
            #2nd part of circle
            fake_summer = gen_S(winter_img)
            disc_summ_fake = disc_S(fake_summer)
            disc_summ_real = disc_S(summer_img)

            summ_fake_loss = mse(disc_summ_fake, torch.zeros_like(disc_summ_fake))
            summ_real_loss = mse(disc_summ_real, torch.ones_like(disc_summ_real))
            summ_loss = summ_fake_loss + summ_real_loss

            total_disc_loss = win_loss + summ_loss

            # logging
            writer.add_scalar('Loss/winter_fake_loss', win_fake_loss, i)
            writer.add_scalar('Loss/winter_real_loss', win_real_loss, i)
            writer.add_scalar('Loss/winter_loss', win_loss, i)
            #
            writer.add_scalar('Loss/summer_fake_loss', summ_fake_loss, i)
            writer.add_scalar('Loss/summer_real_loss', summ_real_loss, i)
            writer.add_scalar('Loss/summer_loss', summ_loss, i)

            writer.add_scalar('Loss/total_disc_loss', total_disc_loss, i)

        #
        optim_disc.zero_grad()
        disc_scaler.scale(total_disc_loss).backward(retain_graph=True)
        disc_scaler.step(optim_disc)
        disc_scaler.update()

        # Train Generators
        

        with torch.cuda.amp.autocast():
            # adversarial loss
            #use imgs from disc train part
            disc_win_fake = disc_W(fake_winter)
            disc_summ_fake = disc_S(fake_summer)
            gen_win_loss = mse(disc_win_fake, torch.ones_like(disc_win_fake))
            gen_summ_loss = mse(disc_summ_fake, torch.ones_like(disc_summ_fake))

            #cycle consistency loss
            gen_winter_img = gen_W(fake_summer)
            gen_summer_img = gen_S(fake_winter)

            winter_loss = l1(winter_img, gen_winter_img)
            summer_loss = l1(summer_img, gen_summer_img)

            total_gen_loss = gen_win_loss + gen_summ_loss + (winter_loss + summer_loss)*config.LAMBDA_CYCLE

            # logging
            writer.add_scalar('Loss/gen_winter_loss', gen_win_loss, i)
            writer.add_scalar('Loss/gen_summer_loss', gen_summ_loss, i)
            #
            writer.add_scalar('Loss/winter_cycle_loss', winter_loss, i)
            writer.add_scalar('Loss/summer_cycle_loss', summer_loss, i)
            #
            writer.add_scalar('Loss/total_gen_loss', total_gen_loss, i)
        
        optim_gen.zero_grad()
        gen_scaler.scale(total_gen_loss).backward()
        gen_scaler.step(optim_gen)
        gen_scaler.update()

        if i % 200 == 0:
            save_image(fake_summer * 0.5 + 0.5, f'saved_train_images/fake_summer_iter_{i}.png') 
            save_image(fake_winter * 0.5 + 0.5, f'saved_train_images/fake_winter_iter_{i}.png')
        

def run_train():
    disc_W = Discriminator().to(config.DEVICE)
    disc_S = Discriminator().to(config.DEVICE)

    gen_W = Generator(img_channels=3).to(config.DEVICE)
    gen_S = Generator(img_channels=3).to(config.DEVICE)

    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    # we make step for both discriminators => have one optim for both discriminators simultaneously
    optim_disc = config.OPTIMIZER(
        list(disc_W.parameters()) + list(disc_S.parameters()),
        lr = config.LEARNING_RATE
    )
    #the same for generators
    optim_gen = config.OPTIMIZER(
        list(gen_W.parameters()) + list(gen_S.parameters()),
        lr = config.LEARNING_RATE
    )

    test_ds = SummerWinterDataset(summer_path=config.TEST_SUMM_DIR,
                                    winter_path=config.TEST_WINT_DIR,
                                    image_size=config.IMAGE_SIZE,
                                    transform=config.TEST_TRANSFORMS)
    test_dl = DataLoader(test_ds,
                         batch_size=config.BATCH_SIZE, 
                         shuffle=False, 
                         num_workers=config.NUM_WORKERS,
                         pin_memory=True)
    
    train_ds = SummerWinterDataset(summer_path=config.TRAIN_SUMM_DIR,
                                    winter_path=config.TRAIN_WINT_DIR,
                                    image_size=config.IMAGE_SIZE,
                                    transform=config.TRAIN_TRANSFORMS)
    train_dl = DataLoader(train_ds,
                         batch_size=config.BATCH_SIZE, 
                         shuffle=True, 
                         num_workers=config.NUM_WORKERS,
                         pin_memory=True)
    
    #
    if config.LOAD_MODEL:
        config.load_cycle_gan_inplace(gen_W=gen_W, gen_S=gen_S,
                                      opt_gen=optim_gen,
                                      disc_W=disc_W, disc_S=disc_S,
                                      opt_disc=optim_disc)

    #there will be a separate subfolder for each launch
    start_time = config.get_time()
    tb_log_dir = os.path.join(config.MAIN_TB_DIR, start_time)
    writer = SummaryWriter(tb_log_dir)

    for epo in range(config.EPOCHS):
        start_time = time.time()

        train(disc_W=disc_W,
              disc_S=disc_S,
              gen_W=gen_W,
              gen_S=gen_S,
              optim_disc=optim_disc,
              optim_gen=optim_gen,
              disc_scaler=disc_scaler,
              gen_scaler=gen_scaler,
              dloader=train_dl,
              writer=writer
            )
        
        elapsed_time = time.time() - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        clear_output(wait=True)
        #<idea> save img's per epo in tensorboard
        print(f'epo {epo}/{config.EPOCHS}')

        print(f'time pre epo {minutes}m {seconds}s')
    # saving
    config.save_cycle_gan(gen_W=gen_W, gen_S=gen_S,
                          opt_gen=optim_gen,
                          disc_W=disc_W, disc_S=disc_S,
                          opt_disc=optim_disc,
                          time_str=start_time)
    show_real_and_fake(gen_W=gen_W, gen_S=gen_S, dataset=test_ds, clr_output=False)

if __name__ == '__main__':
    run_train()