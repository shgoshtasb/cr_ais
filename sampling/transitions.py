import torch
import torch.nn as nn
import numpy as np

from utils.aux import repeat_data, binary_crossentropy_stable, binary_crossentropy_logits_stable
'''
 Parts of the code taken from MCVAE github repository
 https://github.com/stat-ml/mcvae/
'''
    
class BaseTransition(nn.Module):
    '''
    Base class for Markov transitions
    '''
    def __init__(self, input_dim, step_size=0.1, update='fixed', n_tune=10, 
                 tune_inc=1.1, target_accept_ratio=0.8, min_step_size=0.001, max_step_size=1.0, gamma_0=0.1, 
                 name='Markov'):
        super(BaseTransition, self).__init__()
        
        self.name = name
        self.input_dim = input_dim
        self.update = update
        self.n_tune = n_tune
        self.tune_inc = tune_inc
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        self.gamma_0 = gamma_0
        self.target_accept_ratio = 0.8
        self.logvar = 2 * torch.Tensor([np.log(step_size)]).reshape(1, 1).repeat(1, input_dim)
        if self.update == 'learn':
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=True)
            self.register_parameter('logvar', self.logvar)
        else:
            self.logvar = torch.nn.Parameter(self.logvar, requires_grad=False)
            self.register_parameter('logvar', self.logvar)

    @property
    def step_size(self):
        return torch.exp(0.5 * self.logvar)
    
    def forward(self, z, log_w, target_log_density, beta, update_step_size=False, n_samples=1):
        if update_step_size and (self.update == 'tune' or self.update == 'grad-std-tune'):
            for i in range(self.n_tune):
                z_ = z.clone()
                log_w_ = log_w.clone()
            
                _, _, logger_dict = self.step(z_, log_w_, target_log_density, beta)
                self.tune_step_size(logger_dict['a'], logger_dict['log_accept_ratio'], logger_dict['grad_std'])
                
        z, log_w, log_w_inc, logger_dict = self.step(z, log_w, target_log_density, beta)
        logger_dict = logger_dict_update(z=z.reshape(z.shape + (1,)), log_w=log_w, **logger_dict)

        return z, log_w, log_w_inc, logger_dict
        
    def tune_step_size(self, a=None, log_accept_ratio=None, grad_std=None):
        step_size = self.step_size
        
        if self.update == 'tune':
            if torch.exp(log_accept_ratio).mean(dim=0) < self.target_accept_ratio:
                step_size /= self.tune_inc
            else:
                step_size *= self.tune_inc

            step_size = torch.clamp(step_size, max=self.max_step_size, min=self.min_step_size)
        elif self.update == 'grad-std-tune':
            step_size = 0.9 * step_size + 0.1 * self.gamma_0 / (grad_std + 1.)
            if torch.exp(log_accept_ratio).mean(dim=0) < self.target_accept_ratio:
                self.gamma_0 /= self.tune_inc
            else:
                self.gamma_0 *= self.tune_inc
            
        self.logvar.data = 2 * torch.log(step_size)

    def step(self, z, log_w, target_log_density, beta):
        pass
    
def MH_accept_reject(log_accept_ratio):
    accept_ratio = torch.exp(log_accept_ratio)
    # torch.rand returns in [0,1]
    # Uniform returns from [0,1) - need to avoid 0s
    u = 1. - torch.distributions.Uniform(0., 1.).sample(log_accept_ratio.shape).to(log_accept_ratio.device)
    a = (u < accept_ratio).to(torch.float32)
    log_accept_prob = log_accept_ratio.clone()
    log_accept_prob[a == 0] = torch.log(1. - accept_ratio[a == 0])
    return a, log_accept_prob

def logger_dict_update(**kwargs):
    logger_dict = {}
    for k in kwargs.keys():
        if kwargs[k] is not None:
            logger_dict[k] = kwargs[k].clone().detach()
    return logger_dict
    
def MH_update(z_prop, z, target_log_density, proposal_log_ratio):
    z_prop_log_density = target_log_density(z_prop)
    z_log_density = target_log_density(z)

    log_accept_ratio = torch.clamp(z_prop_log_density - z_log_density - proposal_log_ratio, max=-0.)
    a, log_accept_prob = MH_accept_reject(log_accept_ratio)

    z = a * z_prop + (1. - a) * z
    new_z_log_density = a * z_prop_log_density + (1. - a) * z_log_density
    return z, a, z_log_density, new_z_log_density, log_accept_ratio, log_accept_prob


class MH(BaseTransition):
    def __init__(self, input_dim, context_dim, hidden_dim, proposal_sampler, proposal_log_density, 
                 symmetric=True, step_size=0.5, update='fixed', 
                 n_tune=1, tune_inc=1.002, target_accept_ratio=0.8, name='MH'):
        super().__init__(input_dim, step_size, update, n_tune, tune_inc, target_accept_ratio, name=name)
        self.context_dim = context_dim
        self.symmetric = symmetric
        self.device = 'cuda'
        self.proposal_sampler = proposal_sampler
        self.proposal_log_density = proposal_log_density
                        
    def step(self, z, log_w, target_log_density, beta):
        u_prop = self.proposal_sampler(z).to(z.device).float() 
        z_prop = z + self.step_size * u_prop
        if self.symmetric:
            proposal_log_ratio = 0.
        else:
            proposal_log_ratio = self.proposal_log_density(u_prop) - self.proposal_log_density(-u_prop)

        z, a, old_z_log_density, z_log_density, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, target_log_density, proposal_log_ratio)
        log_w = log_w + old_z_log_density - z_log_density
        return z, log_w, old_z_log_density - z_log_density, {'a':a, 'log_accept_ratio':log_accept_ratio, 'log_accept_prob':log_accept_prob}
                
            
def get_grad_z_log_density(log_density, z):
    flag = z.requires_grad
    if not flag:
        z_ = z.detach().requires_grad_(True)
    else:
        z_ = z.requires_grad_(True)
    with torch.enable_grad():
        s = log_density(z_)
        grad = torch.autograd.grad(s.sum(), z_, create_graph=True, only_inputs=True, allow_unused=True)[0]
        if not flag:
            grad = grad.detach()
            z_.requires_grad_(False)
        return grad


class HMC(BaseTransition):
    def __init__(self, input_dim, momentum_sampler, momentum_log_density, step_size, update='fixed', n_tune=1, 
                 tune_inc=1.002, target_accept_ratio=0.65, partial_refresh=200, alpha=1., n_leapfrogs=1, name='HMC'):
        super().__init__(input_dim, step_size, update, n_tune, tune_inc, target_accept_ratio, name=name)
        self.momentum_sampler = momentum_sampler
        self.momentum_log_density = momentum_log_density
        self.register_buffer('n_leapfrogs', torch.tensor(n_leapfrogs))
        self.register_buffer('partial_refresh', torch.tensor(partial_refresh, dtype=torch.int32))
        self.partial_refresh_ = self.partial_refresh.cpu().numpy()
        self.register_buffer('alpha', torch.tensor(alpha))

    def multileapfrog(self, z, p, target_log_density):
        grad_std = []
        z_ = z
        grad = get_grad_z_log_density(target_log_density, z_)
        grad_std.append(torch.std(grad, dim=0, keepdim=True))
        p_ = p + self.step_size / 2. * grad
        for i in range(self.n_leapfrogs):
            z_ = z_ + self.step_size * p_
            grad = get_grad_z_log_density(target_log_density, z_)
            grad_std.append(torch.std(grad, dim=0, keepdim=True))
            if i % self.partial_refresh_ == 0 and i < self.n_leapfrogs - 1:
                p_ = p_ + self.step_size / 2. * grad
                p_ = p_ * self.alpha + torch.sqrt(1. - self.alpha**2) * self.momentum_sampler(p_.shape).to(z.device)
                p_ = p_ + self.step_size / 2. * grad                
            elif i < self.n_leapfrogs - 1:
                p_ = p_ + self.step_size * grad
        p_ = p_ + self.step_size / 2. * grad
        grad_std = torch.cat(grad_std, dim=0)
        grad_std = grad_std.mean(dim=0, keepdim=True)
        return z_, p_, grad_std
    
    def step(self, z, log_w, target_log_density, beta):
        p = self.momentum_sampler(z.shape).to(z.device)
        z_prop, p_prop, grad_std = self.multileapfrog(z, p, target_log_density)
        proposal_log_ratio = self.momentum_log_density(p) - self.momentum_log_density(p_prop)
        z, a, old_z_log_density, z_log_density, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, target_log_density, proposal_log_ratio)
        log_w = log_w + old_z_log_density - z_log_density
        return (z, log_w, old_z_log_density - z_log_density, {'a':a, 'log_accept_ratio':log_accept_ratio, 'grad_std':grad_std, 'log_accept_prob':log_accept_prob, 'up_top':old_z_log_density, 'down_below':z_log_density})


class MALA(BaseTransition):
    def __init__(self, input_dim, step_size, update='fixed', n_tune=1, tune_inc=1.002, target_accept_ratio=0.8, name='MALA'):
        super().__init__(input_dim, step_size, update, n_tune, tune_inc, target_accept_ratio, name=name)

    def step_(self, z, log_w, target_log_density, beta):
        eps = torch.randn_like(z).to(z.device)
        grad = get_grad_z_log_density(target_log_density, z)
        grad_std = torch.std(grad, dim=0)
        u = self.step_size * grad + torch.sqrt(2 * self.step_size) * eps
        z_prop = z + u
        
        rev_grad = get_grad_z_log_density(target_log_density, z_prop)
        reverse_eps = (-u - self.step_size * rev_grad) / torch.sqrt(2. * self.step_size)
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                        scale=torch.tensor(1., device=z.device, dtype=torch.float32))
        proposal_log_ratio = std_normal.log_prob(eps).sum(dim=-1, keepdim=True) - std_normal.log_prob(reverse_eps).sum(dim=-1, keepdim=True)
        z, a, old_z_log_density, z_log_density, log_accept_ratio, log_accept_prob = MH_update(z_prop, z, target_log_density, proposal_log_ratio)
        log_w = log_w + old_z_log_density - z_log_density
        
        return (z, log_w, old_z_log_density - z_log_density, {'a':a, 'log_accept_ratio':log_accept_ratio, 'grad_std':grad_std, 'log_accept_prob':log_accept_prob})
           

class ULA(BaseTransition):
    def __init__(self, input_dim, step_size, update='fixed', n_tune=1, tune_inc=1.002, target_accept_ratio=0.8, name='ULA'):
        super().__init__(input_dim, step_size, update, n_tune, tune_inc, target_accept_ratio, name=name)

    def step(self, z, log_w, target_log_density, beta):
        eps = torch.randn_like(z).to(z.device)
        grad = get_grad_z_log_density(target_log_density, z)
        grad_std = torch.std(grad, dim=0)
        u = self.step_size * grad + torch.sqrt(2 * self.step_size) * eps
        z_prop = z + u
        
        rev_grad = get_grad_z_log_density(target_log_density, z_prop)
        reverse_eps = (-u - self.step_size * rev_grad) / torch.sqrt(2. * self.step_size)
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                        scale=torch.tensor(1., device=z.device, dtype=torch.float32))
        proposal_log_ratio = std_normal.log_prob(eps).sum(dim=-1, keepdim=True) - std_normal.log_prob(reverse_eps).sum(dim=-1, keepdim=True)
        ## for logging purposes 
        _, a, _, _, log_accept_ratio, log_accept_prob = MH_update(z_prop.clone(), z.clone(), target_log_density, proposal_log_ratio)
        z = z_prop
        log_w = log_w - proposal_log_ratio
                
        return (z, log_w, - proposal_log_ratio, {'a':a, 'log_accept_ratio':log_accept_ratio, 'grad_std':grad_std, 'log_accept_prob':log_accept_prob})
        

class ULA_S(BaseTransition):
    def __init__(self, input_dim, step_size, score, update='fixed', n_tune=1, tune_inc=1.002, target_accept_ratio=0., name='ULA'):
        super().__init__(input_dim, step_size, update, n_tune, tune_inc, target_accept_ratio, name=name)
        self.score = score

    def step(self, z, log_w, target_log_density, beta):
        eps = torch.randn_like(z).to(z.device)
        grad = get_grad_z_log_density(target_log_density, z)
        grad_std = torch.std(grad, dim=0)
        u = self.step_size * grad + torch.sqrt(2 * self.step_size) * eps
        z_prop = z + u
        
        rev_grad = get_grad_z_log_density(target_log_density, z_prop)
        reverse_eps = (-u - self.step_size * rev_grad + 2 * self.step_size * self.score(z_prop, beta)) / torch.sqrt(2. * self.step_size)
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., device=z.device, dtype=torch.float32),
                                                        scale=torch.tensor(1., device=z.device, dtype=torch.float32))
        proposal_log_ratio = std_normal.log_prob(eps).sum(dim=-1, keepdim=True) - std_normal.log_prob(reverse_eps).sum(dim=-1, keepdim=True)
        ## for logging purposes 
        _, a, _, _, log_accept_ratio, log_accept_prob = MH_update(z_prop.clone(), z.clone(), target_log_density, proposal_log_ratio)
        z = z_prop
        log_w = log_w - proposal_log_ratio
                
        return (z, log_w, - proposal_log_ratio, {'a':a, 'log_accept_ratio':log_accept_ratio, 'grad_std':grad_std, 'log_accept_prob':log_accept_prob})
    