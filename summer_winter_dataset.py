from typing import Any
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class SummerWinterDataset(Dataset):
    def __init__(self, summer_path, winter_path, image_size = 256, transform = None):
        self.summer_path = summer_path
        self.winter_path = winter_path

        self.image_size = image_size

        self.summer_imgs_names = os.listdir(summer_path)
        self.winter_imgs_names = os.listdir(winter_path)

        self.len = max(len(self.summer_imgs_names), len(self.winter_imgs_names))
        self.summer_len = len(self.summer_imgs_names)
        self.winter_len = len(self.winter_imgs_names)

        if transform is None:
            transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],), 
                            transforms.Resize(self.image_size, antialias=True)
                        ])
        self.transform = transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, indx):
        summer_img_name = self.summer_imgs_names[indx % self.summer_len]
        winter_img_name = self.winter_imgs_names[indx % self.winter_len]

        summer_img_full_path =  os.path.join(self.summer_path, summer_img_name)
        winter_img_full_path =  os.path.join(self.winter_path, winter_img_name)

        summer_img = self.transform(Image.open(summer_img_full_path).convert("RGB"))
        winter_img = self.transform(Image.open(winter_img_full_path).convert("RGB"))

        return summer_img, winter_img
    
    def param_info(self):
        info_str = (f'path to summer images: {self.summer_path}\n'+
            f'path to winter images: {self.winter_path}\n'
            f'transform {str(self.transform)}\n' +
            f'length of summer images: {self.summer_len}\n' + 
            f'length of winter images: {self.winter_len}')
        print(info_str)




class ContitionalPrint:
    def __init__(self, dont_print):
        self.dont_print = dont_print
    def __call__(self, *args, **kwds):
        if not self.dont_print:
            print(*args, **kwds)


def test_1(summer_path, winter_path, silent_mode = False):
    '''
    testing simple iteration
    with default path:
        summer_path = 'data/test_summer'
        winter_path = 'data/test_winter'
    '''
    
    cprint = ContitionalPrint(dont_print=silent_mode)

    # craeting Dataset
    try:
        
        sw_ds = SummerWinterDataset(summer_path=summer_path,
                                    winter_path=winter_path,
                                    image_size = 256,
                                    transform = None)
    except Exception as e:
        print('Unable to create SummerWinterDataset')
        #it seems usles if there is an exception we will able to detect error place by traising
        raise

    # testing __len__ and __getitem__
    cprint('-'*10)
    cprint('call __len__:\n', len(sw_ds))
    cprint('-'*10)
    cprint('call shape of __getitem__ and indx=0:\n', sw_ds[0][0].shape, sw_ds[0][1].shape)
    cprint('-'*10)
    # check param_info
    try:
        sw_ds.param_info()
    except Exception as e:
        print('Exception from "SummerWinterDataset" from "param_info" method')
        raise
        
    cprint('-'*10)
    try:
        for i, (img1, img2) in enumerate(sw_ds):
            cprint(f"{i}/{len(sw_ds)}", end='\r')
        cprint('iteration test - success')
    except Exception as e:
        print('Unable to iterate')
        # it seems lb usless again
        raise

def test_2(summer_path, winter_path , n=5,):
    '''
    show n first images from dataset
    defoul n = 5
    '''

    # if suddenly test_1 wasnt called  
    try:
        sw_ds = SummerWinterDataset(summer_path=summer_path,
                                    winter_path=winter_path,
                                    image_size = 256,
                                    transform = None)
    except Exception as e:
        print('Unable to create SummerWinterDataset')
        #it seems usles if there is an exception we will able to detect error place by traising
        raise
    
    # show 2xn grid of images
    #first line - summer imgs, and second winter
    plt.figure(figsize=(12, 6))
    for i, (summer_img, winter_img) in enumerate(sw_ds):
        if i >= 5:
            break

        #I assume that that image mean and std both equals 0.5
        mean = std = 0.5
        summer_img = (summer_img  * std) + mean
        winter_img = (winter_img  * std) + mean


        plt.subplot(2, n, i+1)
        plt.axis("off")
        plt.imshow(summer_img.permute(1, 2, 0))
        plt.title('Summer in Yosemite')

        plt.subplot(2, n, i+6)
        plt.axis("off")
        plt.imshow(winter_img.permute(1, 2, 0))
        plt.title('Winter in Yosemite')
    plt.show()

if __name__ == '__main__':
    print('start testing summer_winter_dataset')

    current_directory = os.getcwd()
    summer_path = os.path.join(current_directory, 'data/test_summer')
    winter_path = os.path.join(current_directory, 'data/test_winter')

    test_1(summer_path, winter_path)
    test_2(summer_path, winter_path)

    print('summer_winter_dataset successfuly passed all tests')
