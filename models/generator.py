from .resnet_block import ResnetBlock
import torch.nn as nn
import torch
from torchsummary import summary
from utils.visualization import*

class Downsample(nn.Module):
    def __init__(self, n_downsampling, ngf=64, norm_layer=nn.BatchNorm2d, activation=None):
        super(Downsample, self).__init__()

        model = []
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out
    
class Upsample(nn.Module):
    def __init__(self, n_downsampling=3, ngf=64, norm_layer=nn.BatchNorm2d, activation=None, output_nc=0) -> None:
        super(Upsample, self).__init__()

        model = []
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
            
        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output

class ResBlock(nn.Module):
    def __init__(self, num_features=64, n_blocks=5, padding_type='reflect', norm_layer=nn.BatchNorm2d, activation = None) -> None:
        super(ResBlock, self).__init__()

        model = []
        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(num_features, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        out = self.model(input)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=6, activation='relu', norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', cfg=None):
        super(Generator, self).__init__()  

        self.config = cfg
        self.debug_mode = cfg.debug_mode
        activation = nn.ReLU(True) if activation == 'relu' else None

        self.input_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            activation
        )
        
        self.down_sample = Downsample(n_downsampling=n_downsampling, ngf=ngf, norm_layer=norm_layer, activation=activation)
        self.res_block = ResBlock(num_features = ngf * (2**n_downsampling), n_blocks=n_blocks, norm_layer=norm_layer, activation=activation)
        self.up_sample = Upsample(n_downsampling=n_downsampling, ngf=ngf, norm_layer=norm_layer, activation=activation, output_nc=output_nc)

        self.output_block = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()  
        )

    
    def forward(self, input, epoch=None, filename=None):
        x = self.input_block(input)
        x = self.down_sample(x) # 512 channels
        x = self.res_block(x)
        if self.debug_mode :
            save_feature_maps(x, os.path.join(self.config.project_dir, 'visualization', 'G_feature_maps'), filename, epoch)
        x = self.up_sample(x)
        output = self.output_block(x)
        return output
        

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=6, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)       

if __name__ == '__main__':
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Generator(input_nc=2, output_nc=3).to(device)
    print(model)
    input_shape = (2, 128, 128)
    summary(model, input_shape)
    
