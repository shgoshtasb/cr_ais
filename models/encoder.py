import torch
import torch.nn as nn
import torchvision
import numpy as np

from .nets import Down, DoubleConv, get_conv2d, get_DCGANdlayer, weights_init

def get_BiGAN_encoder_net(net, data_shape, latent_dim, n_device):
    if net == 'wu-wide':
        #encoder_net = WuFCEncoder1(data_shape, latent_dim)
        encoder_net = BiGANEncoder(data_shape, latent_dim)
    elif net == 'dcgan':
        encoder_net = DCGANEncoder(data_shape[1],
                                        data_shape[0], latent_dim=latent_dim)
    else:
        raise NotImplemented
    if n_device > 0:
        encoder_net.cuda()
    if n_device > 1:
        encoder_net = nn.DataParallel(encoder_net, range(n_device))
    return encoder_net

def get_encoder_net(net, data_shape, latent_dim, n_device):
    if net == 'wu-wide':
        encoder_net = WuFCEncoder1(data_shape, latent_dim)
    elif net == 'wu-small':
        encoder_net = WuFCEncoder2(data_shape, latent_dim)
    elif net == 'wu-shallow':
        encoder_net = WuFCEncoder3(data_shape, latent_dim)
    elif net == 'binary':
        encoder_net = BinaryEncoder(data_shape, latent_dim)
    elif net == 'conv':
        encoder_net = ConvEncoder(nn.GELU, latent_dim,
                                        data_shape[0], data_shape[1])
    elif net == 'dcgan':
        encoder_net = DCGANEncoder(data_shape[1],
                                        data_shape[0], latent_dim= latent_dim)
    else:
        raise NotImplemented
    if n_device > 0:
        encoder_net.cuda()
    if n_device > 1:
        encoder_net = nn.DataParallel(encoder_net, range(n_device))
    return encoder_net

# Wide fc encoder architecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCEncoder1(nn.Module):
    def __init__(self, data_shape, latent_dim=50):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 1024), nn.Tanh(), nn.Dropout(0.2),
                             nn.Linear(1024, 1024), nn.Tanh(), nn.Dropout(0.2),
                             nn.Linear(1024, 1024), nn.Tanh(), nn.Dropout(0.2),
                             nn.Linear(1024, latent_dim))

    def forward(self, x):
        return self.net(x)
    
# Small fc encoder achitecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCEncoder2(nn.Module):
    def __init__(self, data_shape, latent_dim=10):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 256), nn.Tanh(),
                             nn.Linear(256, 64), nn.Tanh(),
                             nn.Linear(64, latent_dim), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

# Wide but shallow fc encoder architecture from [Huang 
# et al https://arxiv.org/abs/2008.06653]    
# they don't have an explicit encoder but I think it's like this
class WuFCEncoder3(nn.Module):
    def __init__(self, data_shape, latent_dim=50):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 1024), nn.Tanh(), nn.Dropout(0.2),
                             nn.Linear(1024, latent_dim))

    def forward(self, x):
        return self.net(x)
    
# Shallow fc encoder architecture used in [Burda et al. 2016
# https://arxiv.org/abs/1509.00519] 
class BinaryEncoder(nn.Module):
    def __init__(self, data_shape, latent_dim=10):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                                nn.Linear(input_dim, 200), nn.Tanh(),
                                nn.Linear(200, 200), nn.Tanh(),
                                nn.Linear(200, latent_dim))
        
    def forward(self, x):
        return self.net(x)

# Conv encoder in https://github.com/stat-ml/mcvae
class ConvEncoder(nn.Module):
    def __init__(self, act_func, hidden_dim, n_channels, shape, upsampling=True):
        super().__init__()
        self.n_channels = n_channels
        self.upsampling = upsampling
        factor = 2 if upsampling else 1
        num_maps = 16
        num_factor = num_maps // factor
        num_units = int((shape // 8) ** 2) * (8 * num_factor)

        self.net = nn.Sequential(  # n
            DoubleConv(int(n_channels), num_maps, act_func),
            Down(num_maps, 2 * num_maps, act_func),
            Down(2 * num_maps, 4 * num_maps, act_func),
            Down(4 * num_maps, 8 * num_factor, act_func),
            nn.Flatten(),
            nn.Linear(num_units, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)

# Conv encoder like DCGAN discriminator like BiGAN https://arxiv.org/abs/1605.09782  
class DCGANEncoder(nn.Module):
    def __init__(self, image_shape=64, nc=3, latent_dim=100, ndf=64, depth=4, normalize='bn', 
                 spectral_norm=False, ngpu=1):
        super().__init__()
        self.ngpu = ngpu
        output_shape = image_shape
        output_nc = ndf
        bias = False if normalize == 'bn' else True
        
        layers = []
        layers.append(get_conv2d(nc, output_nc, bias, spectral_norm))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        output_shape = int(output_shape / 2.)
        for i in range(depth - 1):
            layers.extend(get_DCGANdlayer(output_shape, output_nc, None, normalize, spectral_norm))
            output_shape = int(output_shape / 2.)
            output_nc *= 2
            
        layers.append(nn.Flatten())
        layers.append(nn.Linear(output_nc * output_shape * output_shape, latent_dim, bias=bias))
        self.net = nn.Sequential(*layers)
        self.apply(weights_init)

        
    def forward(self, x):
        return self.net(x)
    
#BiGAN encoder like BiGAN https://arxiv.org/abs/1605.09782 with one extra middle block
class BiGANEncoder(nn.Module):
    def __init__(self, data_shape, latent_dim=50):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Flatten(),
                             nn.Linear(input_dim, 1024), nn.LeakyReLU(0.2),
                             nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
                             nn.Linear(1024, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2),
                             nn.Linear(1024, latent_dim))

    def forward(self, x):
        return self.net(x)
