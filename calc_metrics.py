import os
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

generated_imgs_folder_path = r'D:\VS_code_proj\CycleGAN\evaluation\generated_summer'
gt_imgs_folder_path = r'D:\VS_code_proj\CycleGAN\evaluation\test_summer'

def get_images_list(path, transform):
    img_t_list = []
    imgs_names = os.listdir(path)
    for single_img_name in imgs_names:
        if not single_img_name.endswith('ini'):
            img_full_path =  os.path.join(path, single_img_name)
            img_t_list.append(transform(Image.open(img_full_path).convert("RGB")))
    
    return torch.stack(img_t_list) 

def calc_IS(generated_images):
    
    #generated_images = get_images_list(generated_imgs_path, transform)
    
    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)

    metric = InceptionScore()
    metric.attach(default_evaluator, "is")
    state = default_evaluator.run([generated_images])
    
    print('IS score is:', state.metrics["is"])
    return

def calc_FID(generated_imgs, gt_images):
    def eval_step(engine, batch):
        return batch
    default_evaluator = Engine(eval_step)

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # comparable metric
    pytorch_fid_metric = FID()#num_features=dims, feature_extractor=wrapper_model)
    pytorch_fid_metric.attach(default_evaluator, "fid")
    state = default_evaluator.run([[generated_imgs, gt_images]])
    print('fid', state.metrics["fid"])

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
    
    #calc_IS(generated_imgs)
    
    calc_FID(generated_imgs[:min_len], gt_images[:min_len])

if __name__ == '__main__':
    #calc_FID()
    calc_metrics()