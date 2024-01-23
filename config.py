'''
config
'''
import torch
import torch.optim as optim
from torchvision import transforms

from datetime import datetime
import warnings
import os
import math

LOAD_MODEL = False
SAVED_MODEL_DIR_PATH = r'saved_models\256\2023_12_01_05h33m'
#r'saved_models\256\2023_12_01_05h33m' - good sw
#r'saved_models\256\2023_12_04_06h57m'
#This is my path to model 



IMAGE_SIZE = 256

NUM_WORKERS = 6 # for my r5-3600 with 6 cores and 12 threads
DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    warnings.warn('The Current deice is cpu it will work slow.', category=UserWarning)

LAMBDA_CYCLE = 10

BATCH_SIZE = 1
EPOCHS = 200
warmup_steps = 30
total_lr_warmup_epochs = EPOCHS

LEARNING_RATE = 0.001
OPTIMIZER = optim.AdamW

dataset_type = 'SW'# 'HZ' # HZ - horse zebra; SW - summer winter
###DIR's###
current_directory = os.getcwd()
#here will be saved state_dict's pre training lauch
CHECK_POINT_DIR = os.path.join(current_directory, 'check_points')

if dataset_type == 'SW':
    TEST_SUMM_DIR = os.path.join(current_directory, r'data\test_summer')
    TEST_WINT_DIR = os.path.join(current_directory, r'data\test_winter')

    TRAIN_SUMM_DIR = os.path.join(current_directory, r'data\train_summer')
    TRAIN_WINT_DIR = os.path.join(current_directory, r'data\train_winter')

elif dataset_type == 'HZ':
    TEST_SUMM_DIR = os.path.join(current_directory, r'horse_zebra_ds\testA')
    TEST_WINT_DIR = os.path.join(current_directory, r'horse_zebra_ds\testB')

    TRAIN_SUMM_DIR = os.path.join(current_directory, r'horse_zebra_ds\trainA')
    TRAIN_WINT_DIR = os.path.join(current_directory, r'horse_zebra_ds\trainB')


MAIN_TB_DIR = os.path.join(current_directory, 'tb_logs')
SAVED_MODELS_DIR = 'saved_models'

### Transforms ##

TEST_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
                            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)
                        ])
TRAIN_TRANSFORMS = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), 
                            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
                            transforms.RandomHorizontalFlip(p=0.5)
                        ])


### func's ###
def warmup_lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos((step - warmup_steps) / float(max(1, (total_lr_warmup_epochs+1) - warmup_steps)) * math.pi)))
## <warmup>



def get_time():
    current_datetime = datetime.now()
    return current_datetime.strftime('%Y_%m_%d_%Hh%Mm')

###func's for saving and loading models###
# The functions save_model and load_checkpoint was taken from: 
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/Pix2Pix/utils.py

def save_checkpoint(model, optimizer, filename):
    
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

def save_cycle_gan(gen_W, gen_S, opt_gen, disc_W, disc_S, opt_disc, time):

    # path looks like .../IMAGE_SIZE/2023_01_01_01h_01m/filename

    ###I was thinking of adding more information about LAMBDA_CYCLE, but decided dont do it
    #current_confi = f'imgs_{IMAGE_SIZE}_lamb_{LAMBDA_CYCLE}'

    time = time.strftime('%Y_%m_%d_%Hh%Mm')
    path = os.path.join(SAVED_MODELS_DIR, str(IMAGE_SIZE))
    path = os.path.join(path, time)

    if not os.path.exists(path):
        os.makedirs(path)
    
    print('saving cyclegan...')
    save_checkpoint(gen_W, opt_gen, filename=os.path.join(path, 'gen_W'))
    save_checkpoint(gen_S, opt_gen, filename=os.path.join(path, 'gen_S'))
    save_checkpoint(disc_W, opt_disc, filename=os.path.join(path, 'disc_W'))
    save_checkpoint(disc_S, opt_disc, filename=os.path.join(path, 'disc_S'))
    print('saved')

def load_cycle_gan_inplace(gen_W, gen_S, opt_gen, disc_W, disc_S, opt_disc, dir_path):
    print('loading model', dir_path)
    load_checkpoint(
        os.path.join(dir_path, 'gen_W'),
        gen_W,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'gen_S'),
        gen_S,
        opt_gen,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'disc_W'),
        disc_W,
        opt_disc,
        LEARNING_RATE,
    )
    load_checkpoint(
        os.path.join(dir_path, 'disc_S'),
        disc_S,
        opt_disc,
        LEARNING_RATE,
    )
    print('model loaded')