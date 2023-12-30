'''
Generator file for CycleGAN
'''
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type = nn.InstanceNorm2d, use_activation=True, **kwargs):
        super().__init__()

        moduls = [
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
            norm_type(out_channels)]
        
        if use_activation:
            moduls.append(nn.ReLU(True))

        self.conv = nn.Sequential(*moduls)

    def forward(self, x):
        return self.conv(x)
    
class ConvTransBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_type = nn.InstanceNorm2d, use_activation=True, **kwargs):
        super().__init__()

        moduls = [
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            norm_type(out_channels)]
        
        if use_activation:
            moduls.append(nn.ReLU(True))
        else:
            moduls.append(nn.Identity())

        self.conv = nn.Sequential(*moduls)

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, norm_type = nn.InstanceNorm2d):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1, norm_type=norm_type),
            ConvBlock(channels, channels, use_activation=False, kernel_size=3, padding=1, norm_type=norm_type),
        )

    def forward(self, x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, n_resblocks=9, norm_type = nn.InstanceNorm2d):
        super().__init__()
        
        self.inital_conv_blocks = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            norm_type(num_features),
            nn.ReLU(True))
        
        self.down_blocks = nn.Sequential(
            ConvBlock(num_features, num_features * 2, norm_type=norm_type, kernel_size=3, stride=2, padding=1),
            ConvBlock(num_features * 2, num_features * 4, norm_type=norm_type, kernel_size=3, stride=2, padding=1,))
        
        res_blocks = []
        for i in range(n_resblocks):
            res_blocks.append(ResidualBlock(num_features * 4, norm_type=norm_type))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up_conv_blocks = nn.Sequential(
            ConvTransBlock(
                    num_features * 4,
                    num_features * 2,
                    norm_type=norm_type,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                ConvTransBlock(
                    num_features * 2,
                    num_features * 1,
                    norm_type=norm_type,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ))
        
        self.last_blocks = nn.Sequential(
            nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect"))
    
    def forward(self, x):
        x = self.inital_conv_blocks(x)
        x = self.down_blocks(x)
        x = self.res_blocks(x)
        x = self.up_conv_blocks(x)
        x = self.last_blocks(x)

        return x
    
#Unfortunately, I havn't come up with a better name
def gen_get_output_and_print_shapes(model, x):
    output_str_length = 20

    names_and_shapes = {'input': x.shape}
    
    x = model.inital_conv_blocks(x)
    names_and_shapes['after inital'] = x.shape

    x = model.down_blocks(x)
    names_and_shapes['after down'] = x.shape

    x = model.res_blocks(x)
    names_and_shapes['after residual'] = x.shape

    x = model.up_conv_blocks(x)
    names_and_shapes['after up'] = x.shape

    x = model.last_blocks(x)
    names_and_shapes['after last'] = x.shape

    for name, shape in names_and_shapes.items():
        print(name.ljust(output_str_length), shape)

    return x.shape

def test_gen_shape(silent_mode=False):
    gen = Generator(img_channels = 3)
    noise = torch.randn(1, 3, 256, 256)

    if silent_mode:
        final_shape = gen(noise).shape
    else:
        final_shape = gen_get_output_and_print_shapes(gen, noise)
    
    assert noise.shape == final_shape, f'Input and ouput from Generator have different shapes. Input shape is {noise.shape}. But output is {final_shape}'
    
def print_layers():
    gen = Generator(img_channels = 3)
    gen_blocks_list = [
        gen.inital_conv_blocks,
        gen.down_blocks,
        gen.res_blocks,
        gen.up_conv_blocks,
        gen.last_blocks
    ]
    layers_names = [
        'inital',
        'downsample',
        'residuals',
        'up convolution',
        'last block'
    ]

    for block, name in zip(gen_blocks_list, layers_names):
        print(name)
        print(block)


if __name__ == '__main__':
    print('start testing generator')

    test_gen_shape()
    print('-'*20)
    #print_layers()

    print('generator successfuly passed all tests')




        
