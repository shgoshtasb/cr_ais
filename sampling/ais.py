import torch
import torch.nn as nn
import torchvision
import numpy as np

from sampling.annealing import get_schedule, get_transition, get_annealing_path, get_f
from utils.aux import repeat_data, get_ess, logselfnormalize, freeze
    
class AIS(torch.nn.Module):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='',
                 annealing_kwargs={'M':128, 'schedule':'geometric', 'path':'geometric'},
                 transition='Neal', transition_kwargs={}, 
                 device=1, logger=None, name='AIS', **kwargs):
        super().__init__()
        self.name = name
        self.ckpt=ckpt
        self.device = 'cuda' if device > 0 else 'cpu'
        self.n_device = device
        self.logger_ = logger
        # Module dimentions
        self.input_dim = input_dim
        self.context_dim = context_dim
        # Proposal and target [unnormalized] log_densities and 
        # proposal sample generator
        self.set_annealing(**annealing_kwargs)
        self.target_log_density = target_log_density
        self.set_context(None)
        self.set_proposal()
        self.set_transition(transition, **transition_kwargs)
        
    def set_annealing(self, schedule, M, path, alpha=0.):
        self.schedule = schedule
        self.M = M
        self.beta_ = None
        self.path = path
        self.alpha = alpha
        
    @property
    def annealing_path(self):
        return get_annealing_path(self.current_proposal_log_prob, 
                                  self.current_target_log_density, self.path, self.alpha)
        
    @property
    def beta(self):
        if self.beta_ is None:
            self.beta_ = get_schedule(self.M, self.schedule).to(self.device)
        return self.beta_
    
    @beta.setter
    def beta(self, value):
        self.beta_ = value
        
    def get_beta(self, i):
        return self.beta[i]

    def get_last_beta(self, device):
        if len(self.beta_) == 0:
            return torch.tensor(0).to(device)
        else:
            return self.beta_[-1]
            
    def set_proposal(self):
        self.register_buffer('zero', torch.zeros(self.input_dim, dtype=torch.float32, requires_grad=False).to(self.device))
        self.register_buffer('one', torch.ones(self.input_dim, dtype=torch.float32, requires_grad=False).to(self.device))
        self.proposal = torch.distributions.Normal(loc=self.zero, scale=self.one)
            
    def set_transition(self, transition, **kwargs):
        self.transitions = get_transition(self.input_dim, self.context_dim, self.M, transition, **kwargs)
        
    def get_context(self, x):
        return x
    
    def set_context(self, x):
        self.current_proposal_log_prob = lambda z: self.proposal.log_prob(z).sum(dim=-1, keepdim=True)
        self.current_proposal_sample = lambda z_samples: self.proposal.sample(torch.Size([z_samples,])).to(self.device)
        self.current_target_log_density = lambda z: self.target_log_density(z, x)
        self.current_rx = x
        self.current_rxembed = None if x is None else self.get_context(x)
            
    def init_sample(self, n_samples=1, x=None):
        if x is not None and x.nelement() == 0:
            x = None
        self.set_context(x if x is None else repeat_data(x, n_samples))
        z_samples = n_samples if self.current_rx is None else self.current_rx.shape[0]
        z = self.current_proposal_sample(z_samples)
        log_probz = self.current_proposal_log_prob(z)
        log_w = -log_probz
        return z, log_w
        
    def sample(self, n_samples=1, x=None, update_step_size=False, log=False):
        z, log_w = self.init_sample(n_samples, x)
        return self(n_samples, z, log_w, update_step_size, log)

    def update_transition_logs(self, z, log_w, transition_logs={}, logger_dict=None, stats=None, log=False):
        if len(transition_logs.keys()) == 0 or not log:
            transition_logs = {
                'z': [z.clone().detach().reshape(z.shape + (1,))],
                'log_w': [log_w.clone().detach()],
                'log_accept_prob': [torch.zeros_like(log_w).clone().detach()],
            }
            if stats:
                for k in stats.keys():
                    transition_logs[k] = [stats[k].reshape(1,).clone().detach()]
        else:
            for k in logger_dict.keys():
                if k not in transition_logs.keys():
                    transition_logs[k] = []
                if k != 'log_w':
                    transition_logs[k].append(logger_dict[k].clone().detach())
            transition_logs['log_w'].append(log_w.clone().detach())
            if stats:
                for k in stats.keys():
                    transition_logs[k].append(stats[k].reshape(1,).clone().detach())
        return transition_logs

    def forward(self, n_samples, z, log_w, update_step_size=False, log=False, transition_logs={}):
        transition_logs = self.update_transition_logs(z, torch.zeros_like(log_w), log=log)
            
        for i in range(self.M):
            z, log_w, _, logger_dict = self.transitions(z, log_w, 
                                             self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)            
            transition_logs = self.update_transition_logs(z, log_w + self.annealing_path(self.get_beta(i))(z), transition_logs, logger_dict, log=log)
                
        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        log_w = log_w + self.current_target_log_density(z)
        transition_logs['logZ'] = torch.logsumexp(log_w.view(n_samples, -1), dim=0) - np.log(n_samples)
        return z, log_w, transition_logs
        
class Plain(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='', 
                 annealing_kwargs={'M':128, 'schedule':'geometric', 'path':'geometric', },
                 transition='RWMH', transition_kwargs={}, 
                 device=1, logger=None, name='Plain', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, ckpt, annealing_kwargs,
                         transition, transition_kwargs, device, logger, name)
        for p in self.parameters():
            p.requires_grad_(False)

class AdaptiveAIS(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='',
                 annealing_kwargs={'path':'geometric', 'max_step':1/8, 'min_step':1e-3, 'conditional':False, 'ratio':0.5},
                 transition='RWMH', transition_kwargs={},
                 device=1, logger=None, name='Adaptive', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, ckpt, annealing_kwargs,
                         transition, transition_kwargs, device, logger, name)
        for p in self.parameters():
            p.requires_grad_(False)
        self.transition_kwargs = transition_kwargs
        self.transition = transition

    def set_annealing(self, path, alpha=0., max_step=1/8, min_step=1e-3, conditional=False, ratio=0.5):
        self.max_step = max_step
        self.min_step = min_step
        self.tune_beta = True
        self.beta_ = []
        self.path = path
        self.alpha = alpha
        self.conditional = conditional
        self.ratio = ratio
        self.M = None
        self.max_M = 131072
                
    def get_ess(self, log_w, log_w_inc):
        if self.conditional:
            log_nw = logselfnormalize(log_w)
            inv = torch.logsumexp(log_nw + 2 * log_w_inc, dim=0) - 2 * torch.logsumexp(log_nw + log_w_inc, dim=0)
            return torch.exp(-inv) * log_w.shape[0]
        else:
            return get_ess(log_w + log_w_inc)
        
    def update_thr(self, ess, n_samples):
        if self.conditional:
            return ess * self.ratio
        else:
            return ess * self.ratio
        
    def init_step(self, beta):
        if 1. - beta < self.min_step:
            return 1. - beta
        else:
            return min(self.max_step, 1. - beta)

    def tune_beta_step(self, n_samples, z, log_w, ess, update_step_size):
        last_beta = self.get_last_beta(z.device)
        step = self.init_step(last_beta)
        thr = self.update_thr(ess, n_samples)

        _, _, log_w_inc, _ = self.transitions(z, log_w, self.annealing_path(last_beta + step), update_step_size)
        ess = self.get_ess(log_w, log_w_inc)

        # Not a full binary search stops the search when beta is small enough to make it less exhaustive
        while ess < thr and step > self.min_step:
            step /= 2.
            _, _, log_w_inc, _ = self.transitions(z, log_w, self.annealing_path(last_beta + step), last_beta + step, update_step_size)
            ess = self.get_ess(log_w, log_w_inc)
        self.beta_.append(last_beta + step)        
        
    def forward(self, n_samples, z, log_w, update_step_size=False, log=False, transition_logs={}):
        transition_logs = self.update_transition_logs(z, torch.zeros_like(log_w), log=log)

        i = 0
        ess = n_samples
        while (self.tune_beta and len(self.beta) < self.max_M and (len(self.beta) == 0 or self.beta[-1] < 1)) or (not self.tune_beta and i < len(self.beta)):
            if self.tune_beta:
                self.tune_beta_step(n_samples, z, log_w, ess, update_step_size)
                
            z, log_w, log_w_inc, logger_dict = self.transitions(z, log_w, self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
            ess = self.get_ess(log_w, log_w_inc)
                
            transition_logs = self.update_transition_logs(z, log_w + self.annealing_path(self.get_beta(i))(z), transition_logs, logger_dict, log=log)
            i += 1
            
        if self.tune_beta:
            self.beta_ = torch.stack(self.beta_)
            self.M = self.beta_.shape[0]

        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        log_w = log_w + self.current_target_log_density(z)
        transition_logs['logZ'] = torch.logsumexp(log_w.view(n_samples, -1), dim=0) - np.log(n_samples)
        return z, log_w, transition_logs

            
class CRAIS(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='',
                 annealing_kwargs={'path':'geometric', 'tol':1e-1, 'max_M':1024, 'dt':1, },
                 transition='RWMH', transition_kwargs={},
                 device=1, logger=None, name='CR', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, ckpt, annealing_kwargs,
                         transition, transition_kwargs, device, logger, name)
        for p in self.parameters():
            p.requires_grad_(False)
        self.transition_kwargs = transition_kwargs
        self.transition = transition

    def set_annealing(self, path, alpha=0., dt=1., max_M=1024, tol=1e-7):
        self.dt = dt
        self.tol = tol
        self.max_M = max_M
        self.tune_beta = True
        self.beta_ = []
        self.path = path
        self.alpha = alpha
        self.max_step = 1/8
        self.beta_clip = True
        self.M = None

    def tune_beta_step(self, V):
        last_beta = self.get_last_beta(V.device)
        
        beta = 1 - (1 - last_beta) * torch.exp(- self.dt / V)
        if self.beta_clip:
            beta = torch.clamp(beta, max=last_beta + self.max_step)
        beta = torch.clamp(beta, max=1.)
        self.beta_.append(beta)
        
    def forward(self, n_samples, z, log_w, update_step_size=False, log=False, transition_logs={}):
        stats = self.get_stats(z, log_w, torch.zeros(1).to(z.device))
        transition_logs = self.update_transition_logs(z, torch.zeros_like(log_w), stats=stats, log=log)
        i = 0
        
        while (self.tune_beta and len(self.beta_) < self.max_M and stats['V'] > self.tol) or (not self.tune_beta and i < len(self.beta_)):
            if self.tune_beta:                   
                self.tune_beta_step(stats['V'])
            z, log_w, _, logger_dict = self.transitions(z, log_w, self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
            
            stats = self.get_stats(z, log_w, self.get_beta(i))
            transition_logs = self.update_transition_logs(z, log_w + self.annealing_path(self.get_beta(i))(z), transition_logs, logger_dict, stats, log=log)
            
            i += 1

        if self.tune_beta:
            if len(self.beta_) == 0 or self.beta_[-1] < 1.:
                self.beta_.append(torch.tensor(1.).to(self.device))

                z, log_w, _, logger_dict = self.transitions(z, log_w, self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
                stats = self.get_stats(z, log_w, self.get_beta(i))            
                transition_logs = self.update_transition_logs(z, log_w, transition_logs, logger_dict, stats, log=log)
                
            self.beta_ = torch.stack(self.beta_)
            self.M = self.beta_.shape[0]

        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        
        log_w = log_w + self.current_target_log_density(z)
        transition_logs['logZ'] = torch.logsumexp(log_w.view(n_samples, -1), dim=0) - np.log(n_samples)
        return z, log_w, transition_logs
        
    def get_stats(self, z, log_w, beta):
        z_target_log_density = self.current_target_log_density(z)
        z_step_log_density = self.annealing_path(beta)(z)

        log_w_target = log_w + z_target_log_density
        log_w_step = log_w + z_step_log_density

        log_Zpi = torch.logsumexp(log_w_target, dim=0) - np.log(z.shape[0])
        log_Zq = torch.logsumexp(log_w_step, dim=0) - np.log(z.shape[0])
        log_Zratio = log_Zpi - log_Zq
        Zratio = torch.exp(log_Zratio)
        
        log_tilde_u = z_target_log_density - z_step_log_density
        log_u = log_tilde_u - log_Zratio
        
        f = get_f(self.alpha, log_u)
        norm_w = torch.exp(logselfnormalize(log_w_step))
        f_mean = (norm_w * f).sum(dim=0)

        if self.alpha == 0:
            Vin = log_tilde_u
        else:
            Vin = torch.exp(self.alpha * (log_tilde_u - .5 * log_Zratio[0]) - np.log(np.abs(self.alpha)))
        V = ((Vin - Vin.mean(dim=0))**2).mean()
        
        return {'f':f_mean, 'V':V}
    