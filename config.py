'''
The functions save_model and load_checkpoint was taken from: 
https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/utils.py
'''

import torch.optim as optim
import torch

IMAGE_SIZE = 256

DEVICE = 'cuda'

LAMBDA_CYCLE = 10

BATCH_SIZE = 1
LEARNING_RATE = 3e-4
OPTIMIZER = optim.AdamW

#here will be saved state_dict's pre training lauch
CHECK_POINT_DIR = 'CycleGAN/check_points'



def save_model(model, optimizer, filename):

    states = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(states, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_cycle_gan(gen1, disc1, optim1, gen2, disc2, optim3):
    pass
def load_cycle_gan():
    pass
