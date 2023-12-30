'''
Discriminator file for CycleGAN

The image size should be 3x256x256.
In this scenario, the Discriminator returns a 30x30 map of probabilities
'''
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type = nn.InstanceNorm2d, use_activation=True, **kwargs):
        super().__init__()
        moduls = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, padding= 1, bias = True, padding_mode="reflect", **kwargs),
            norm_type(out_channels)]
        
        if use_activation:
            moduls.append(nn.LeakyReLU(0.2, inplace=True))

        self.conv = nn.Sequential(*moduls)

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, n_features = [64, 128, 256, 512], use_sigmoid=True):
        super().__init__()
        self.init_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels= n_features[0],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect'
                      ),
            nn.LeakyReLU(0.2, inplace=True)
        )

        middle_layers = []
        #iterating for pairs of i and i+1 elements except the last one
        for in_ch, out_ch in zip(n_features[:-2], n_features[1:-1]):
            middle_layers.append(
                ConvBlock(in_channels=in_ch,
                          out_channels=out_ch,
                          stride = 2)
            )
        middle_layers.append(
            ConvBlock(in_channels=n_features[-2],
                      out_channels=n_features[-1],
                      stride = 1)
        )
        self.middle_block = nn.Sequential(*middle_layers)

        last_block_list = []
        last_block_list.append(
            nn.Conv2d(
                in_channels=n_features[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode='reflect'
            )
        )
        if use_sigmoid:
            last_block_list.append(nn.Sigmoid())
        
        self.last_block = nn.Sequential(*last_block_list)

    def forward(self, x):
        x = self.init_block(x)
        x = self.middle_block(x)
        return self.last_block(x)

def test_shapes():
    '''
    print shape of intput after every layer block
    '''
    discr = Discriminator()
    noise = torch.randn(1, 3, 256, 256)
    names_and_shapes = {'input': noise.shape}

    noise = discr.init_block(noise)
    names_and_shapes['after initial block'] = noise.shape
    
    noise = discr.middle_block(noise)
    names_and_shapes['after middle block'] = noise.shape

    noise = discr.last_block(noise)
    names_and_shapes['after last block'] = noise.shape

    output_str_length = 20
    for name, shape in names_and_shapes.items():
        print(name.ljust(output_str_length), shape)
    
    #Check for output size is not required to allow for random input image sizes

if __name__ == '__main__':
    print('start testing discriminator')

    print('-'*20)
    test_shapes()
    print('-'*20)

    print('discriminator successfuly passed all tests')
    
    

