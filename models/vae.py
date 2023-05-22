import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.aux import log_normal_density, repeat_data, binary_crossentropy_logits_stable
from .decoder import Base, get_decoder_net
from .encoder import get_encoder_net

# We use a similar approach to [Huang et al. 2020 https://arxiv.org/abs/2008.06653]
# and ignore the gaussian observation model in VAE for equivalence with GANs 
# (d(x, f(z)) = |x - f(z)|^2)
class VAE(Base):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='wu-wide', learning_rate=[1e-4], 
                 logger=None, device=2, likelihood="bernoulli", dataset="mnist", name="VAE"):
        self.net = net
        self.dc = 2 if likelihood == 'normal' else 1
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         learning_rate, logger, device, likelihood, dataset, name)
            
    def get_nets(self):
        self.get_encoder()
        self.get_decoder()
    
    def get_optimizers(self):
        self.optimizers = [optim.Adam(list(self.encoder_net.parameters()) + \
                                      list(self.decoder_net.parameters()), 
                                      lr=self.learning_rate[0], betas=(0.5, 0.999))]
        self.schedulers = []
        
    def get_encoder(self):
        self.encoder_net = get_encoder_net(self.net, self.data_shape, 2 * self.latent_dim, self.device)
                                   
    def get_decoder(self):
        output_shape = self.data_shape * np.array([self.dc, 1, 1])
        self.decoder_net = get_decoder_net(self.net, output_shape, self.latent_dim, self.device)
    
    def encode(self, x):
        output = self.encoder_net(x)
        mean, log_var = output[:, :self.latent_dim], output[:, self.latent_dim:]
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        eps = torch.randn_like(mean).to(mean.device)
        z = mean + eps * torch.exp(0.5 * log_var)
        log_q = log_normal_density(z, mean, log_var)
        return z, log_q

    def decode(self, z):
        x_recon = self.decoder_net(z)[:, :self.data_shape[0]]
        return x_recon

    def decoder_with_var(self, z):
        output = self.decoder_net(z)
        x_recon, x_log_var = output[:, :self.data_shape[0]], output[:, self.data_shape[0]:]
        if x_log_var.shape == ([]):
            x_log_var = None
        return x_recon, x_log_var
                                                
    def loss_fn(self, z, x, log_joint_density, log_q, n_samples=1):
        log_w = (log_joint_density - log_q).reshape(n_samples, -1)
        # IWAE loss from [Burda et al. https://arxiv.org/abs/1509.00519]
        if n_samples > 1:
            if self.training:
                log_q = log_q.view(n_samples, -1)
                log_joint_density = log_joint_density.view(n_samples, -1)
                log_w = log_joint_density - log_q
                #for stability
                log_sw = log_w - torch.max(log_w, dim=0)[0]
                w = torch.exp(log_sw)
                w /= w.sum(dim=0)
                return -(w.detach() * log_w).sum(dim=0).mean()
            else:
                return -torch.logsumexp(log_w, dim=0).mean() + np.log(n_samples)
        else:
            return -log_w.mean()
                
    def step(self, x, n_samples=1, opt_idx=0):
        mean, log_var = self.encode(x)
        mean = mean.repeat(n_samples, 1)
        log_var = log_var.repeat(n_samples, 1)
        z, log_q = self.reparameterize(mean, log_var)
        
        x = repeat_data(x, n_samples)
        x_recon, x_log_var = self.decoder_with_var(z)    
        if x_log_var is None:
            x_log_var = self.log_var                                            
        log_joint_density = self.log_joint_density(z, x, x_recon, x_log_var)
        nelbo = self.loss_fn(z, x, log_joint_density, log_q, n_samples)
        return {'nelbo': (nelbo, 1.)}

# Implementation of [Burda et al. 2016 https://arxiv.org/abs/1611.04273]     
class IWAE(VAE):
    def __init__(self, data_shape, latent_dim, deep, log_var=np.log(2.), 
                 log_var_trainable=False, net='binary', learning_rate=[1e-4], 
                 logger=None, device=2, likelihood="bernoulli", dataset="mnist", name="IWAE"):
        super().__init__(data_shape, latent_dim, deep, log_var, log_var_trainable, 
                         net, learning_rate, logger, device, likelihood, 
                         dataset, name)

    def get_optimizers(self):
        self.optimizers = [optim.Adam(list(self.encoder_net.parameters()) + \
                                      list(self.decoder_net.parameters()), 
                                      lr=self.learning_rate[0])]
        self.schedulers = [optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 12, 39, 120, 363, 1092, 3279], gamma=(0.1)**(1./7)) for optimizer in self.optimizers]
        