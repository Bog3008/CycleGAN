import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import timm
from ignite.metrics import FID, InceptionScore
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


def get_images_list(path, transform):
    img_t_list = []
    imgs_names = os.listdir(path)
    for single_img_name in imgs_names:
        if not single_img_name.endswith('ini'):
            img_full_path =  os.path.join(path, single_img_name)
            img_t_list.append(transform(Image.open(img_full_path).convert("RGB")))
    
    return torch.stack(img_t_list) 

def calc_IS(generated_images):
    
    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)

    metric = InceptionScore()
    metric.attach(default_evaluator, "is")
    state = default_evaluator.run([generated_images])
    
    #print('IS score is:', state.metrics["is"])
    return state.metrics["is"]

def calc_FID(generated_imgs, gt_images):
    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # comparable metric
    pytorch_fid_metric = FID()
    pytorch_fid_metric.attach(default_evaluator, "fid")
    state = default_evaluator.run([[generated_imgs, gt_images]])
    
    #print('fid', state.metrics["fid"])
    return state.metrics["fid"]

def calc_metrics():
    # Calculate FID and Inception Score

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    generated_imgs = get_images_list(generated_imgs_folder_path, transform)
    gt_images = get_images_list(gt_imgs_folder_path, transform)
    min_len = min(len(generated_imgs), len(gt_images)) # number of examples in FID for y_pred and y_gt must be the same  
    
    IS = calc_IS(generated_imgs)
    FID = calc_FID(generated_imgs[:min_len], gt_images[:min_len])

    print('IS:', IS)
    print('FID:', FID)

if __name__ == '__main__':
    if len(sys.argv) != 3: #file name, path to generated images and path to gt images
        raise RuntimeError('Incorrect number of arguments. You must specify only path to generated images folder and to gt images folder')  
    generated_imgs_folder_path = sys.argv[1]
    gt_imgs_folder_path = sys.argv[2]
    print('start calculating IS and FID')
    calc_metrics()

# summer
#D:\VS_code_proj\CycleGAN\evaluation\generated_summer
#D:\VS_code_proj\CycleGAN\evaluation\test_summer
#winter
#D:\VS_code_proj\CycleGAN\evaluation\generated_win
#D:\VS_code_proj\CycleGAN\evaluation\test_winter

#D:/Anaconda/python.exe d:/VS_code_proj/CycleGAN/CycleGAN/calc_metrics.py 
'''
Summer
IS: 3.505470456649775
FID: 0.11375601871979296

Winter
IS: 2.7693440524808484
FID: 0.09692788565606697'''