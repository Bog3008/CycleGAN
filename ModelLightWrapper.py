from generator import Generator
import config
import torch
import os
from PIL import Image
from torchvision.utils import save_image

class GeneratorWrapper:
    def __init__(self, transformations_paths):
        '''
        transformations_paths - dictionary like {'horse2zebra':r'somepath/saved_models/horse2zebra',...}
        '''
        self.generator = Generator(img_channels=3).to(config.DEVICE)
        self.transformations_paths = transformations_paths

        self.current_transformation = None
    def __call__(self, transormation, image_path):
        '''
        transformation - smth like 'horse2zebra' or 'zebra2horse'
        '''
        if transormation not in ['horse2zebra', 'zebra2horse', 'summer2winter', 'winter2summer']:
            raise NotImplementedError
        
        if not os.path.exists(image_path):
            raise RuntimeError(f'the path to image doesnt exist. You set{image_path}')
        
        if transormation != self.current_transformation:
            self.load_model(transormation)
        
        image = Image.open(image_path)
        image = config.TEST_TRANSFORMS(image)

        generated_image = self.generator(torch.unsqueeze(image.to(config.DEVICE, dtype=torch.float), 0))
        generated_image = generated_image[0].cpu()

        std=mean=0.5
        generated_image = generated_image * std + mean
        save_image(torch.cat((image, generated_image), dim=0), image_path)
        

    def load_model(self, transormation):
        try:
            saved_model_path = self.transformations_paths[transormation]
        except:
            raise RuntimeError(f'transformations_paths(arg in constructor) doesnt contains transormation as a key. transformation you have sent{transormation}')
        
        if not os.path.exists(saved_model_path):
            raise RuntimeError(f'the path you set in transformations_paths doesnt exist in your environment. You set{saved_model_path}')
        
        #loading model
        checkpoint = torch.load(saved_model_path, map_location=config.DEVICE)
        self.generator.load_state_dict(checkpoint["model"])