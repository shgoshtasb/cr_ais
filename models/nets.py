import torch
import torch.nn as nn

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
def get_normalizer_module(normalize, shape):
    if normalize == 'bn':
        return nn.BatchNorm2d(shape[0])
    elif normalize == 'in':
        return nn.InstanceNorm2d(shape[0])
    elif normalize == 'ln':
        return nn.LayerNorm(shape)
    else:
        return Identity()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0., .02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1., .02)
        m.bias.data.fill_(0.)
        
def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, act_func, mid_channels=None, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            act_func(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            act_func()
        )

    def forward(self, x):
        conv = self.double_conv(x)
        if self.skip_connection and x.shape[1] <= conv.shape[1]:
            return pad_channels(x, conv.shape[1]) + conv
        else:
            return conv


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, act_func):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, act_func)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, act_func, upsampling=True, size=None):
        super().__init__()

        # if upsampling, use the normal convolutions to reduce the number of channels
        if upsampling:
            if size is None:
                self.up = nn.Upsample(scale_factor=2, mode='nearest')  # , align_corners=True
            else:
                self.up = nn.Upsample(size=size, mode='nearest')  # align_corners=True
            self.conv = DoubleConv(in_channels, out_channels, act_func, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, act_func)

    def forward(self, x):
        x = self.conv(self.up(x))
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

        
              
# DCGAN discriminator https://arxiv.org/abs/1511.06434
def get_conv2d(in_channels, out_channels, bias, spectral_norm):
    conv = nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(conv)
    else:
        return conv
    
def get_DCGANdlayer(in_shape, in_channels, out_channels=None, 
                    normalize='bn', spectral_norm=False):
    if out_channels is None:
        out_channels = in_channels * 2
    
    bias = False if normalize == 'bn' else True
    
    layers = []        
    layers.append(get_conv2d(in_channels, out_channels, bias,
                             spectral_norm))
    if normalize is not None:
        layers.append(get_normalizer_module(normalize, 
                        (out_channels, in_shape, in_shape)))   
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers
    
    