'''
config
'''
import torch
import torch.optim as optim
from torchvision import transforms

import warnings
import os

IMAGE_SIZE = 256

NUM_WORKERS = 6 # for my r5-3600 with 6 cores and 12 threads
DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    warnings.warn('The Current deice is cpu it will work slow.', category=UserWarning)

LAMBDA_CYCLE = 10

BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 3e-4
OPTIMIZER = optim.AdamW

TEST_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
                            transforms.Resize(IMAGE_SIZE, antialias=True)
                        ])
TRAIN_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
                            transforms.Resize(IMAGE_SIZE, antialias=True),
                            transforms.RandomHorizontalFlip(p=0.5)
                        ])

###DIR's###
current_directory = os.getcwd()
#here will be saved state_dict's pre training lauch
CHECK_POINT_DIR = os.path.join(current_directory, 'check_points')

TEST_SUMM_DIR = os.path.join(current_directory, 'data/test_summer')
TEST_WINT_DIR = os.path.join(current_directory, 'data/test_winter')

TRAIN_SUMM_DIR = os.path.join(current_directory, 'data/train_summer')
TRAIN_WINT_DIR = os.path.join(current_directory, 'data/train_winter')

###func's for saving and loading models###
# The functions save_model and load_checkpoint was taken from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/utils.py

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
