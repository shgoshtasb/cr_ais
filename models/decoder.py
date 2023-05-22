import torch
import torch.nn as nn
import torchvision
import numpy as np

from utils.aux import log_normal_density, repeat_data, binary_crossentropy_stable
from .nets import get_normalizer_module, weights_init, Up, OutConv

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape
        
    def forward(self, x):
        batch_size = x.shape[0]
        shape = (batch_size, *self.shape)
        return x.view(shape)


# We use a similar approach to [Huang et al. 2020 https://arxiv.org/abs/2008.06653]
# and ignore the gaussian observation model in VAE for equivalence with GANs 
# (d(x, f(z)) = |x - f(z)|^2)
class Base(nn.Module):
    def __init__(self, data_shape, latent_dim, deep, log_var=-np.log(2.), 
                 log_var_trainable=False, learning_rate=[], logger=None, device=2, likelihood="bernoulli", 
                 dataset="mnist", name="Base"):
        super(Base, self).__init__()
        self.data_shape = data_shape
        self.latent_dim = latent_dim
        self.deep = deep
        self.learning_rate = learning_rate
        self.likelihood = likelihood
        self.dataset = dataset
        self.name = name
        self.logger = logger
        self.device = device
        self.current_epoch = 0
        self.best_epoch = 0 
        self.best_loss = np.inf
        
        self.get_nets()
        self.get_optimizers()
        
        self.log_var = nn.Parameter(torch.ones((1,) + tuple(self.data_shape.numpy())) * log_var, requires_grad=log_var_trainable)
        
    def get_nets(self):
        self.get_decoder()
        
    def get_decoder(self):
        pass
    
    def get_optimizers(self):
        pass
    
    def decode(self, z):
        x_recon = self.decoder_net(z)
        return x_recon
        #zeros = torch.zeros_like(z, device=z.device)
        #log_joint_density = lambda x: log_normal_density(z, zeros, zeros) + self.get_loglikelihood(x, x_recon, self.log_var)
        #return x_recon, log_joint_density

    def sample(self, n_samples, log_var=None, noisy=False, device='cuda'):
        zeros = torch.zeros(self.latent_dim, dtype=torch.float32, requires_grad=False).to(device)
        latent_dist = torch.distributions.Normal(loc=zeros, scale=zeros + 1.)
        z = latent_dist.sample(torch.Size([n_samples,]))
        x_gen = self(z)
        if noisy:
            if self.likelihood == "bernoulli":
                observation = torch.distributions.bernoulli.Bernoulli(probs=x_gen.view(x_gen.shape[0], -1))
            elif self.likelihood == "normal":
                observation = torch.distributions.Normal(loc=x_gen.view(x_gen.shape[0], -1), 
                                                         scale=torch.exp(0.5 * log_var).reshape((1,) + log_var.shape))
            x_gen = observation.sample().reshape((x_gen.shape[0],) + self.data_shape)
        return z, x_gen
            
    def log_joint_density(self, z, x, x_recon, log_var):
        zeros = torch.zeros_like(z, device=z.device)
        if x_recon is None:
            x_recon = self(z)
        return log_normal_density(z, zeros, zeros) + self.get_loglikelihood(x, x_recon, log_var)
    
    def get_loglikelihood(self, x, x_recon, log_var):
        x_flat = x.view(x.shape[0], -1)
        x_recon_flat = x_recon.view(x_recon.shape[0], -1)
        if self.likelihood == "bernoulli":
            return -binary_crossentropy_stable(x_recon_flat, x_flat).sum(dim=-1, keepdim=True)
            #return nn.BCELoss(x_recon_flat, x_flat)
        elif self.likelihood == "normal":
            log_var_flat = log_var.view(-1, x_flat.shape[1])
            return log_normal_density(x_flat, x_recon_flat, log_var_flat)
        else:
            raise NotImplemented
            
    def forward(self, z):
        return self.decode(z)
    
    def train_epoch(self, train_loader, n_samples):
        self.train()
        train_losses = {}
        samples = 0
        losses = [0.] * len(self.optimizers)
        for batch_idx, batch in enumerate(train_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            for opt_idx, optimizer in enumerate(self.optimizers):
                #opt_idx = batch_idx % len(self.optimizers)
                optimizer = self.optimizers[opt_idx]
                optimizer.zero_grad()

                loss, loss_dict = self.stepnlog(x, n_samples, opt_idx, log='train')
                losses[opt_idx] += loss.item() * batch_size
                for k in loss_dict.keys():
                    if k in train_losses.keys():
                        train_losses[k] += loss_dict[k][0].item() * batch_size
                    else:
                        train_losses[k] = loss_dict[k][0].item() * batch_size
                loss.backward()
                optimizer.step()            
            samples += batch_size

        for scheduler in self.schedulers:
            scheduler.step()
        for opt_idx in range(len(self.optimizers)):
            losses[opt_idx] /= samples
        for k in train_losses.keys():
            train_losses[k] /= samples
        self.current_epoch += 1
        return losses, train_losses
        
    def validation_epoch(self, val_loader, n_samples):
        self.eval()
        val_losses = {}
        samples = 0
        losses = 0.
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                x, _ = batch
                x = x.cuda()
                batch_size = x.shape[0]

                loss, loss_dict = self.stepnlog(x, n_samples, None, log='val')
                losses += loss.item() * batch_size
                for k in loss_dict.keys():
                    if k in val_losses.keys():
                        val_losses[k] += loss_dict[k][0].item() * batch_size
                    else:
                        val_losses[k] = loss_dict[k][0].item() * batch_size
                samples += batch_size    
            losses /= samples
            for k in val_losses.keys():
                val_losses[k] /= samples
        return losses, val_losses

    def save_gen(self, z, x_gen, normalized, rows, save_dir='./', prefix=None, save=False):
        #if self.dataset in ['cifar10', 'celeba']:
        if x_gen is None:
            x_gen = self(z).clamp(0., 1.)
        #else:
        #    x_gen = torch.sigmoid(self(z))
        #if normalized is not None:
        #    mean, std = normalized
        #    x_gen = x_gen * std.reshape(1, -1, 1, 1).to(z.device) + mean.reshape(1, -1, 1, 1).to(z.device)
        grid = torchvision.utils.make_grid(x_gen, rows, 2)
        img = torchvision.transforms.ToPILImage()(grid)
        if save:
            if prefix is None:
                img.save(f'{save_dir}/image_{model.current_epoch}.jpg')
                print(f'image_{model.current_epoch}.jpg')
            else:
                img.save(f'{save_dir}/image_{prefix}.jpg')
                print(f'image_{prefix}.jpg')
        else:
            # display(img)
            pass
        return x_gen, img
    
    def stepnlog(self, x, n_samples=1, opt_idx=0, log=None):
        loss = 0.
        loss_dict = self.step(x, n_samples, opt_idx)
        if log is not None and self.logger is not None:
            for k in loss_dict.keys():
                self.logger.add_scalar(f'{log}_{k}', 
                                       loss_dict[k][0], self.current_epoch)
        for k in loss_dict.keys():
            loss += loss_dict[k][0] * loss_dict[k][1]
        return loss, loss_dict
        
    def step(self, x, n_samples, opt_idx):
        pass
    
def get_decoder_net(net, output_shape, latent_dim, n_device):
    if net == 'wu-wide':
        decoder_net = WuFCDecoder1(output_shape, latent_dim)
    elif net == 'wu-small':
        decoder_net = WuFCDecoder2(output_shape, latent_dim)
    elif net == 'wu-shallow':
        decoder_net = WuFCDecoder3(output_shape, latent_dim)
    elif net == 'binary':
        decoder_net = BinaryDecoder(output_shape, latent_dim)
    elif net == 'conv':
        decoder_net = ConvDecoder(nn.GELU, latent_dim,
                                       output_shape[0], output_shape[1])
    elif net == 'dcgan':
        decoder_net = DCGANGenerator(output_shape[1], output_shape[0],
                                          latent_dim)
    else:
        raise NotImplemented
    if n_device > 0:
        decoder_net.cuda()
    if n_device > 1:
        decoder_net = nn.DataParallel(decoder_net, range(n_device))
    return decoder_net

# Wide fc decoder architecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCDecoder1(nn.Module):
    def __init__(self, data_shape, latent_dim=50):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Linear(latent_dim, 1024), nn.Tanh(),
                                 nn.Linear(1024, 1024), nn.Tanh(),
                                 nn.Linear(1024, 1024), nn.Tanh(),
                                 nn.Linear(1024, input_dim), nn.Sigmoid(),
                                 View(data_shape))
        
    def forward(self, z):
        return self.net(z)

# Small fc decoder architecture used in [Wu et al. 2017
# https://arxiv.org/abs/1611.04273] 
class WuFCDecoder2(nn.Module):
    def __init__(self, data_shape, latent_dim=10):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Linear(latent_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 256), nn.Tanh(),
                                 nn.Linear(256, 256), nn.Tanh(),
                                 nn.Linear(256, 1024), nn.Tanh(),
                                 nn.Linear(1024, input_dim), nn.Sigmoid(),
                                 View(data_shape))
        
    def forward(self, z):
        return self.net(z)
    
# Wide but shallow fc decoder architecture from [Huang 
# et al https://arxiv.org/abs/2008.06653]    
class WuFCDecoder3(nn.Module):
    def __init__(self, data_shape, latent_dim=50):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Linear(latent_dim, 1024), nn.Tanh(),
                                 nn.Linear(1024, input_dim), nn.Sigmoid(),
                                 View(data_shape))
        
    def forward(self, z):
        return self.net(z)
    
    
# Shallow fc decoder architecture used in [Burda et al. 2016   
# https://arxiv.org/abs/1611.04273] 
class BinaryDecoder(nn.Module):
    def __init__(self, data_shape, latent_dim=10):
        super().__init__()
        input_dim = torch.prod(data_shape).int().numpy()
        self.net = nn.Sequential(nn.Linear(latent_dim, 200), nn.Tanh(),
                                 nn.Linear(200, 200), nn.Tanh(),
                                 nn.Linear(200, input_dim), nn.Sigmoid(),
                                 View(data_shape))
        
    def forward(self, z):
        return self.net(z)

# DCGAN generator https://arxiv.org/abs/1511.06434

def get_convt2d(in_channels, out_channels, bias, spectral_norm):
    convt = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=bias)
    if spectral_norm:
        return nn.utils.spectral_norm(convt)
    else:
        return convt
    
def get_DCGANglayer(in_shape, in_channels, out_channels=None,
                    normalize='bn', spectral_norm=False):
    if out_channels is None:
        out_channels = in_channels // 2
        
    bias = False if normalize == 'bn' else True
    
    layers = []        
    layers.append(get_convt2d(in_channels, out_channels, bias,
                              spectral_norm))
    if normalize is not None:
        layers.append(get_normalizer_module(normalize, 
                    (out_channels, in_shape, in_shape)))     
    layers.append(nn.ReLU(inplace=True))
    return layers
        

class DCGANGenerator(nn.Module):
    def __init__(self, image_shape=64, nc=3, nz=100, ngf=64, depth=4, normalize='bn', 
                 spectral_norm=False):
        super().__init__()
        self.image_shape = image_shape
        output_shape = image_shape
        output_nc = ngf
        bias = False if normalize == 'bn' else True
        
        layers = []
        layers.append(nn.Tanh())
        #layers.append(nn.Sigmoid())
        layers.append(get_convt2d(output_nc, nc, bias, spectral_norm))
        
        for i in range(depth - 1):
            layers.extend(get_DCGANglayer(output_shape, output_nc * 2, None, normalize, 
                                         spectral_norm)[::-1])
            output_shape = int(np.ceil(output_shape / 2.))
            output_nc *= 2
            
        layers.append(nn.ReLU(inplace=True))
        if normalize is not None:
            layers.append(get_normalizer_module(normalize, 
                                            (output_nc, output_shape, output_shape)))
        layers.append(View((output_nc, output_shape, output_shape)))
        layers.append(nn.Linear(nz, output_nc * output_shape * output_shape, bias=bias))
        
        self.layers = layers[::-1]
        self.net = nn.Sequential(*self.layers)
        self.apply(weights_init)

        # DCGANGenerator(
        # (net): Sequential(
        #     (0): Linear(in_features=100, out_features=4096, bias=False)
        #     (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (2): ReLU(inplace)
        #     (3): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (4): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (5): ReLU(inplace)
        #     (6): ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (7): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (8): ReLU(inplace)
        #     (9): ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (10): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (11): ReLU(inplace)
        #     (12): ConvTranspose2d(128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (13): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        #     (14): ReLU(inplace)
        #     (15): ConvTranspose2d(64, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        #     (16): Tanh()
        # )
        # )

    def forward(self, z):
        output = self.net(z)
        if output.shape[2] > self.image_shape:
            output = output[:, :, :self.image_shape, :self.image_shape]
        output = (output + 1) / 2.
        return output
        
# Conv decoder in https://github.com/stat-ml/mcvae        
class ConvDecoder(nn.Module):
    def __init__(self, act_func, hidden_dim, n_channels, shape, upsampling=True):
        super().__init__()
        self.upsampling = upsampling
        factor = 2 if upsampling else 1
        num_maps = 16
        shape_init = shape
        shape = shape // 8
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, (8 * num_maps // factor) * int(shape ** 2)),
            View(((8 * num_maps // factor), shape, shape)),
            Up(8 * num_maps // factor, 4 * num_maps // factor, act_func, upsampling, (shape * 2, shape * 2)),
            Up(4 * num_maps // factor, 2 * num_maps // factor, act_func, upsampling, (shape * 4, shape * 4)),
            Up(2 * num_maps // factor, num_maps, act_func, upsampling, (shape_init, shape_init)),
            OutConv(num_maps, int(n_channels))
        )
        self.hidden_dim = hidden_dim

    def forward(self, z):
        return self.net(z)
