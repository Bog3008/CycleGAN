from generator import Generator
import config
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image
import sys

def modify_image(image_path, saved_model_path, path_to_save):
    generator = Generator(img_channels=3).to(config.DEVICE)


    if not os.path.exists(image_path):
        raise RuntimeError(f'the path to image doesnt exist. You set {image_path}')
    
    if not os.path.exists(saved_model_path):
        raise RuntimeError(f'the path you set in transformations_paths doesnt exist in your environment. You set{saved_model_path}')
    
    #loading model
    checkpoint = torch.load(saved_model_path, map_location=config.DEVICE)
    generator.load_state_dict(checkpoint["model"])
    
    image = Image.open(image_path)

    image = config.TEST_TRANSFORMS(image)
    image = torch.unsqueeze(image.to(config.DEVICE, dtype=torch.float), 0)

    generated_image = generator(image)
    #generated_image = generated_image[0]#.cpu()

    std=mean=0.5
    #generated_image = generated_image * std + mean
    path_to_save += 'generated.png'
    save_image(torch.cat((image, generated_image), dim=0)* std + mean, path_to_save)    
    

if __name__ == '__main__':

    if len(sys.argv) != 4: #file name, path to generated images and path to gt images
        raise RuntimeError('Incorrect number of arguments. You must specify only path the image and path to model and path to generated image to save')  
    img_path = sys.argv[1]
    model_path = sys.argv[2]
    path_to_save = sys.argv[3]
    print('starting...')
    modify_image(img_path, model_path, path_to_save)
    print(f'Done! Chaeck {path_to_save+"generated.png"}')

# D:/Anaconda/python.exe d:/VS_code_proj/CycleGAN/CycleGAN/inference.py D:\VS_code_proj\CycleGAN\CycleGAN\saved_test_images\fake_summer_iter_0.png saved_models\256\2023_12_01_05h33m\gen_W D:\