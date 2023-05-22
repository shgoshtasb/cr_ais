import torch
import torch.nn as nn
import numpy as np

    
class BatchNorm(nn.Module):
    def __init__(self, input_dim, momentum=0.1, epsilon=1e-5):
        super().__init__()
        self.input_dim = input_dim
        self.log_gamma = nn.Parameter(torch.zeros(1, input_dim))
        self.beta = nn.Parameter(torch.zeros(1, input_dim))
        self.momentum = momentum
        self.epsilon = epsilon
        self.register_buffer("moving_mean", torch.zeros((1, input_dim), dtype=torch.float32))
        self.register_buffer("moving_variance", torch.ones((1, input_dim), dtype=torch.float32))

    @property
    def gamma(self):
        return torch.exp(self.log_gamma)

    def forward(self, x):
        out = (x - self.beta) / self.gamma * torch.sqrt(self.moving_variance + self.epsilon) + self.moving_mean
        return out

    def _inverse(self, y):
        if self.training:
            mean = y.mean(dim=0, keepdim=True)
            var = y.var(dim=0, keepdim=True)
            with torch.no_grad():
                self.moving_mean.mul_(1 - self.momentum).add_(mean * self.momentum)
                self.moving_variance.mul_(1 - self.momentum).add_(var * self.momentum)
        else:
            mean = self.moving_mean
            var = self.moving_variance
        out = (y - mean) * self.gamma / torch.sqrt(var + self.epsilon) + self.beta
        return out

    def log_abs_det_jacobian(self, x, y):
        if self.training:
            var = torch.var(y, dim=0, keepdim=True)
        else:
            var = self.moving_variance
        out = (-self.log_gamma + 0.5 * torch.log(var + self.epsilon)).sum(dim=-1, keepdim=True)
        return out
    
                        

class Bijective(nn.Module):
    def __init__(self):
        super(Bijective, self).__init__()
        
    def forward(self, x, context, log_J=None, mode='direct'):
        if log_J is None:
            log_J = torch.zeros(x.shape[0], 1).to(x.device)
        if mode == 'direct':
            return self.forward_(x, context, log_J)
        else:
            return self.backward_(x, context, log_J)
        
    def forward_(self, x, context, log_J):
        pass
    
    def backward_(self, x, context, log_J):
        pass

class Flow(Bijective):
    def __init__(self, input_dim, device):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.register_buffer('zero', torch.zeros(input_dim, dtype=torch.float32, requires_grad=False).to(self.device))
        self.register_buffer('one', torch.ones(input_dim, dtype=torch.float32, requires_grad=False).to(self.device))
    
    def sample(self, n_samples, x=None):
        base_dist = torch.distributions.Normal(loc=self.zero, scale=self.one)
        eps = base_dist.sample(n_samples)
        eps_log_prob = base_dist.log_prob(eps).sum(dim=-1, keepdim=True)
        #eps = torch.randn(n_samples, self.input_dim)
        #eps_log_prob = torch.distributions.Normal(torch.tensor(0., dtype=torch.float32, devi), torch.tensor(1., dtype=torch.float32, device="cuda")).log_prob(eps).sum(dim=-1, keepdim=True)
        z, log_J, _ = self(eps, x)
        return z
    
    def log_prob(self, z, x=None):
        base_dist = torch.distributions.Normal(loc=self.zero, scale=self.one)
        eps, log_J, _ = self(z, x, mode='inverse')
        eps_log_prob = base_dist.log_prob(eps).sum(dim=-1, keepdim=True)
        #eps_log_prob = torch.distributions.Normal(torch.tensor(0., dtype=torch.float32, device="cuda"), torch.tensor(1., dtype=torch.float32, device="cuda")).log_prob(eps).sum(dim=-1, keepdim=True)

        return eps_log_prob + log_J

        
class Permute(Bijective):
    def __init__(self, input_dim, permutation):
        super().__init__()
        self.input_dim = input_dim
        self.register_buffer('permutation', torch.tensor(permutation, dtype=torch.long, requires_grad=False))
        self.register_buffer('inverse_permutation', torch.tensor(np.argsort(permutation), dtype=torch.long, requires_grad=False))

    def forward_(self, x, context, log_J):
        return torch.index_select(x, 1, self.permutation), log_J, 0.
    
    def backward_(self, x, context, log_J):
        return torch.index_select(x, 1, self.inverse_permutation), log_J, 0.

def get_fc(input_dim, hidden_dims, output_dim, activations):
    net = []
    hidden_dims = [input_dim] + hidden_dims + [output_dim]
    if len(activations) < len(hidden_dims) - 1:
        activations.append(None)
    for dim_1, dim_2, act in zip(hidden_dims[:-1], hidden_dims[1:], activations):
        net.append(nn.Linear(dim_1, dim_2))
        if act is not None:
            net.append(act)
    return nn.Sequential(*net)

class AffineCoupling(Bijective):
    def __init__(self, input_dim, split_dim, hidden_dims, context_dim, t_activations, s_activations):
        super().__init__()
        self.split_dim = split_dim
        output_dim = input_dim - split_dim
        self.t_net = get_fc(split_dim + context_dim, hidden_dims, output_dim, t_activations)
        self.s_net = get_fc(split_dim + context_dim, hidden_dims, output_dim, s_activations)
            
    
    def forward_(self, x, context, log_J):
        x1, x2 = x.split(self.split_dim, dim=-1)
        if context is None:
            t = self.t_net(x1)
            logs = self.s_net(x1)
        else:
            t = self.t_net(torch.cat([x1, context], dim=1))
            logs = self.s_net(torch.cat([x1, context], dim=1))
        x = torch.cat([x1, t + torch.exp(logs) * x2], dim=1)
        log_J_inc = logs.sum(dim=-1, keepdim=True)
        log_J = log_J + log_J_inc
        return x, log_J, log_J_inc
        
    def backward_(self, x, context, log_J):
        x1, x2 = x.split(self.split_dim, dim=-1)
        if context is None:
            t = self.t_net(x1)
            logs = self.s_net(x1)
        else:
            t = self.t_net(torch.cat([x1, context], dim=1))
            logs = self.s_net(torch.cat([x1, context], dim=1))
        x = torch.cat([x1, (x2 - t) / torch.exp(logs)], dim=1)
        log_J_inc =  - logs.sum(dim=-1, keepdim=True)     
        log_J = log_J + log_J_inc
        return x, log_J, log_J_inc

class RealNVP(Flow):
    def __init__(self, input_dim, context_dim=0, hidden_dim=20, n_blocks=1, permute=False, batchnorm=False, device='cuda'):
        super().__init__(input_dim, device)
        split_dim = input_dim // 2
        hidden_dims = [hidden_dim] * 2
        t_activations = [nn.LeakyReLU(0.01)] * 2
        s_activations = [nn.LeakyReLU(0.01)] * 2 + [nn.Tanh()]
        
        self.transforms = nn.ModuleList()
        for i in range(n_blocks):
            self.transforms.append(AffineCoupling(input_dim, split_dim, hidden_dims, context_dim, t_activations, s_activations))
            if permute:
                c = (i % 2) * split_dim + ((i + 1) % 2) * (input_dim - split_dim)
                permutation = np.concatenate([np.arange(c, input_dim), np.arange(c)], axis=0)
                self.transforms.append(Permute(input_dim, permutation))
            if batchnorm:
                self.transforms.append(BatchNorm(input_dim, momentum=0.2))
        self.register_buffer('zero', torch.zeros(input_dim, dtype=torch.float32, requires_grad=False))
        self.register_buffer('one', torch.ones(input_dim, dtype=torch.float32, requires_grad=False))
        self.base_dist = torch.distributions.Normal(loc=self.zero, scale=self.one)
    
        for module in self.transforms.parameters():
            if isinstance(module, torch.nn.Linear):
        #        #torch.nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0.)
   
    
    def forward_(self, x, context, log_J):
        log_J_inc = torch.zeros(x.shape[0], 1).to(x.device)
        for i, module in enumerate(self.transforms):
            x, log_J, log_J_inc_ = module(x, context, log_J, mode='direct')
            log_J_inc += log_J_inc_
        return x, log_J, log_J_inc
    
    def backward_(self, x, context, log_J):
        log_J_inc = torch.zeros(x.shape[0], 1).to(x.device)
        for i, module in enumerate(reversed(self.transforms)):
            x, log_J, log_J_inc_ = module(x, context, log_J, mode='inverse')
            log_J_inc += log_J_inc_
        return x, log_J, log_J_inc

    
class Gaussian(Flow):
    def __init__(self, input_dim, mean=None, logvar=None, trainable=True, device="cuda"):
        super().__init__(input_dim, device)
        if mean is None:
            self.mean_ = torch.nn.Parameter(torch.zeros(input_dim).to(self.device), requires_grad=trainable)
            self.register_parameter('mean', self.mean_)
        else:
            self.mean = mean
        if logvar is None:
            self.logvar_ = torch.nn.Parameter(torch.zeros(input_dim).to(self.device), requires_grad=trainable)
            self.register_parameter('logvar', self.logvar_)
        else:
            self.var = var        

    def forward_(self, z, x, log_J):
        if x is None:
            mean = self.mean
            logvar = self.logvar
        else:
            mean = self.mean(x)
            logvar = self.logvar(x)
        scale = torch.exp(.5 * logvar)
        z = mean + scale * z
        log_J_inc = 0.5 * logvar.sum(dim=-1, keepdim=True)
        log_J = log_J + log_J_inc
        return z, log_J, log_J_inc

    def backward_(self, z, x, log_J):
        if x is None:
            mean = self.mean
            logvar = self.logvar
        else:
            mean = self.mean(x)
            logvar = self.logvar(x)
        scale = torch.exp(.5 * logvar)
        z = (z - mean) / scale
        log_J_inc =  - 0.5 * logvar.sum(dim=-1, keepdim=True)
        log_J = log_J + log_J_inc
        return z, log_J, log_J_inc
