'''
training for CycleGAN
'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from summer_winter_dataset import SummerWinterDataset
from generator import Generator
from discriminator import Discriminator
import config

from tqdm import tqdm

def train(disc_W, disc_S,
          gen_W, gen_S,
          optim_disc, optim_gen,
          disc_scaler, gen_scaler,
          dloader):
    
    mse = nn.MSELoss()
    l1 = nn.L1Loss()

    for i, (summer_img, winter_img) in enumerate(tqdm(dloader, leave=False)):
        
        summer_img = summer_img.to(config.DEVICE)
        winter_img = winter_img.to(config.DEVICE)

        # Train Discriminators
        optim_disc.zero_grad()
        
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
        #
        disc_scaler.scale(total_disc_loss).backward()
        disc_scaler.step(optim_disc)
        disc_scaler.update()

        # Train Generators
        optim_gen.zero_grad()

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

            total_gen_loss = (gen_win_loss + gen_summ_loss) + (winter_loss + summer_loss) * config.
        
        gen_scaler.scale(total_gen_loss).backward()
        gen_scaler.step(optim_gen)
        gen_scaler.update()

        if i % 200 == 0:
            save_image(fake_summer * 0.5 + 0.5, f'saved_train_image/summer_iter_{i}.png') 
            save_image(fake_winter * 0.5 + 0.5, f'saved_train_image/winter_iter_{i}.png')
        

def run_train():
    disc_W = Discriminator().to(config.DEVICE)
    disc_S = Discriminator().to(config.DEVICE)

    gen_W = Generator().to(config.DEVICE)
    gen_S = Generator().to(config.DEVICE)

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
                         shuffle=True, 
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
    
    for epo in config.EPOCHS:
        train(disc_W=disc_W,
              disc_S=disc_S,
              gen_W=gen_W,
              gen_S=gen_S,
              optim_disc=optim_disc,
              optim_gen=optim_gen,
              disc_scaler=disc_scaler,
              gen_scaler=gen_scaler,
              dloader=train_dl
            )