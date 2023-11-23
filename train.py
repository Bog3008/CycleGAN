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
from datetime import datetime
import time
import warnings

import matplotlib.pyplot as plt

        

def show_real_and_fake(gen_W, gen_S, dataset, epo=0, save_not_show=True, n_imgs=7):

    std = mean = 0.5
    gen_W.eval()
    gen_S.eval()
    plt.figure(figsize=(20, 20))
    
    for i in range(n_imgs):

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
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plt.subplot(4, n_imgs, i+1)
            plt.axis("off")
            plt.imshow(real_summ.permute(1, 2, 0))
            plt.title('Real Summer')

            plt.subplot(4, n_imgs, i+n_imgs+1)
            plt.axis("off")
            plt.imshow(fake_win.permute(1, 2, 0))
            plt.title('Fake winter')

            plt.subplot(4, n_imgs, i+n_imgs*2+1)
            plt.axis("off")
            plt.imshow(real_win.permute(1, 2, 0))
            plt.title('Real Winter')

            plt.subplot(4, n_imgs, i+n_imgs*3+1)
            plt.axis("off")
            plt.imshow(fake_summ.permute(1, 2, 0))
            plt.title('Fake Summer')

    if save_not_show:
        plt.savefig(f'img_per_epo/img_epo_{epo}_imgs_{config.IMAGE_SIZE}_lamb_{config.LAMBDA_CYCLE}.png')
    else:
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
            
            save_image(torch.cat((winter_img, fake_summer), dim=0) * 0.5 + 0.5, f'saved_train_images/fake_summer_iter_{i}.png') 
            save_image(torch.cat((summer_img, fake_winter), dim=0) * 0.5 + 0.5, f'saved_train_images/fake_winter_iter_{i}.png')
        

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
    
    # uncomment 2 lines down if you wanna use sheduler
    #start_from_epoh_num = 115
    #sheduler_disc = torch.optim.lr_scheduler.LambdaLR(optim_disc, lr_lambda=config.lr_lambda(start_from_epoh_num))
    
    #the same for generators
    optim_gen = config.OPTIMIZER(
        list(gen_W.parameters()) + list(gen_S.parameters()),
        lr = config.LEARNING_RATE
    )
    #sheduler_gen = torch.optim.lr_scheduler.LambdaLR(optim_gen, lr_lambda=config.lr_lambda(start_from_epoh_num))


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
        seved_model_dir_path = config.SAVED_MODEL_DIR_PATH #r'saved_models\256\2023_11_18_19h50m' # move to config
        config.load_cycle_gan_inplace(gen_W=gen_W, gen_S=gen_S,
                                      opt_gen=optim_gen,
                                      disc_W=disc_W, disc_S=disc_S,
                                      opt_disc=optim_disc,
                                      dir_path=seved_model_dir_path)


    #there will be a separate subfolder for each launch


    tb_log_dir = os.path.join(config.MAIN_TB_DIR, config.get_time())
    writer = SummaryWriter(tb_log_dir)

    launch_time = start_time = datetime.now()

    for epo in range(config.EPOCHS):
        print(f'epo {epo+1}/{config.EPOCHS}')

        start_time = datetime.now()

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
        #sheduler_disc.step()
        #sheduler_gen.step()
        
        elapsed_time = datetime.now() - start_time
        minutes = int(elapsed_time.total_seconds() // 60)
        seconds = int(elapsed_time.total_seconds() % 60)
        
        #<idea> save img's per epo in tensorboard
        show_real_and_fake(gen_W=gen_W, gen_S=gen_S, dataset=test_ds, epo=epo)
        os.system('cls')
        
        print(f'time pre epo {minutes}m {seconds}s')

        if epo % 5 == 0:
            config.save_cycle_gan(gen_W=gen_W, gen_S=gen_S,
                          opt_gen=optim_gen,
                          disc_W=disc_W, disc_S=disc_S,
                          opt_disc=optim_disc,
                          time=launch_time)
        

        
    # saving
    #show_real_and_fake(gen_W=gen_W, gen_S=gen_S, dataset=test_ds, save_not_show=False)
    config.save_cycle_gan(gen_W=gen_W, gen_S=gen_S,
                          opt_gen=optim_gen,
                          disc_W=disc_W, disc_S=disc_S,
                          opt_disc=optim_disc,
                          time=start_time)


def test_model(num_to_saved=100, use_train=False):
    print('start_testing...')
    gen_W = Generator(img_channels=3).to(config.DEVICE)
    gen_S = Generator(img_channels=3).to(config.DEVICE)
    
    gen_W.eval()
    gen_S.eval()

    seved_model_dir_path = config.SAVED_MODEL_DIR_PATH
    
    checkpoint = torch.load(os.path.join(seved_model_dir_path, 'gen_W'), map_location=config.DEVICE)
    gen_W.load_state_dict(checkpoint["model"])

    checkpoint = torch.load(os.path.join(seved_model_dir_path, 'gen_S'), map_location=config.DEVICE)
    gen_S.load_state_dict(checkpoint["model"])

    test_ds = SummerWinterDataset(summer_path=config.TEST_SUMM_DIR,
                                    winter_path=config.TEST_WINT_DIR,
                                    image_size=config.IMAGE_SIZE,
                                    transform=config.TEST_TRANSFORMS)
    if use_train == True:
        test_ds = SummerWinterDataset(summer_path=config.TRAIN_SUMM_DIR,
                                    winter_path=config.TRAIN_WINT_DIR,
                                    image_size=config.IMAGE_SIZE,
                                    transform=config.TEST_TRANSFORMS) ## apply test transfmormation to evaluate amodel in correct way
    
    #show_real_and_fake(gen_W=gen_W, gen_S=gen_S, dataset=test_ds, epo=0, save_not_show=False)
    #show_real_and_fake(gen_W=gen_W, gen_S=gen_S, dataset=train_ds, save_not_show=False, epo=0)

    for i, (summer_img, winter_img) in enumerate(test_ds):
        if i >= num_to_saved:
            return
        summer_img = summer_img.to(config.DEVICE, dtype=torch.float)
        winter_img = winter_img.to(config.DEVICE, dtype=torch.float)

        summer_img = torch.unsqueeze(summer_img, 0)
        winter_img = torch.unsqueeze(winter_img, 0)
        
        with torch.no_grad():
            fake_summ = gen_S(winter_img)
            fake_win = gen_W(summer_img)
        fake_summ = fake_summ#[0]#.cpu()
        fake_win = fake_win#[0]#.cpu()
        #print(winter_img.device, summer_img.device)
        #print(fake_summ.device, fake_win.device)
        
        save_image(torch.cat((winter_img, fake_summ), dim=0) * 0.5 + 0.5, f'saved_test_images/fake_summer_iter_{i}.png') 
        save_image(torch.cat((summer_img, fake_win), dim=0) * 0.5 + 0.5, f'saved_test_images/fake_winter_iter_{i}.png')
        
        os.system('cls')
        print(f'Processed {1+i}/{min(num_to_saved, len(test_ds))}')

    print('Done! Check "saved_test_images" foldef')



if __name__ == '__main__':
    #run_train()
    test_model(use_train=True)