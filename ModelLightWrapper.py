from generator import Generator
import config
import torch
import os
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import save_image

class GeneratorWrapper:
    def __init__(self, transformations_paths):
        '''
        transformations_paths - dictionary like {'horse2zebra':r'somepath/saved_models/horse2zebra',...}
        '''
        self.generator = Generator(img_channels=3).to(config.DEVICE)
        self.transformations_paths = transformations_paths

        self.transformation = None
    def __call__(self, transformation, image_path):
        '''
        transformation - smth like 'horse2zebra' or 'zebra2horse'
        '''
        if not os.path.exists(image_path):
            raise RuntimeError(f'the path to image doesnt exist. You set {image_path}')
        
        if transformation != self.transformation:
            self.transformation = transformation
        
        self.load_model(self.transformation)
        
        image = Image.open(image_path)

        image = config.TEST_TRANSFORMS(image)
        image = torch.unsqueeze(image.to(config.DEVICE, dtype=torch.float), 0)

        generated_image = self.generator(image)
        #generated_image = generated_image[0]#.cpu()

        std=mean=0.5
        #generated_image = generated_image * std + mean
        
        save_image(torch.cat((image, generated_image), dim=0)* std + mean, image_path)

        return image_path
        

    def load_model(self, transormation):
        if transormation not in self.transformations_paths.keys():
            raise NotImplementedError(f'The transformation {transormation} not in transformations_paths.keys()')

        try:
            saved_model_path = self.transformations_paths[transormation]
        except:
            raise RuntimeError(f'transformations_paths(arg in constructor) doesnt contains transormation as a key. transformation you have sent{transormation}')
        
        if not os.path.exists(saved_model_path):
            raise RuntimeError(f'the path you set in transformations_paths doesnt exist in your environment. You set{saved_model_path}')
        
        #loading model
        checkpoint = torch.load(saved_model_path, map_location=config.DEVICE)
        self.generator.load_state_dict(checkpoint["model"])