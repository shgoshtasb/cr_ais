import os, pickle
import torch
import numpy as np

from sampling.transitions import MH, HMC, MALA, ULA, ULA_S

def annealing_path_geometric(log_density1, log_density2):
    return lambda beta: lambda z: (1 - beta) * log_density1(z) + beta * log_density2(z)

def annealing_path_linear_(beta, log_density1, log_density2):
    if beta == 0:
        return log_density1
    elif beta == 1.:
        return log_density2
    else:
        return lambda z: torch.logsumexp(torch.cat([ \
                    torch.log(1. - beta + 1e-7) + log_density1(z), 
                    torch.log(beta + 1e-7) + log_density2(z)], dim=1), dim=1, 
                                            keepdim=True)

def annealing_path_linear(log_density1, log_density2):
    return lambda beta: annealing_path_linear_(beta, log_density1, log_density2)

def log_power_mean(z, beta, alpha, log_density1, log_density2):
    if alpha == 0:
        return annealing_path_geometric(log_density1, log_density2)(beta)(z)
    else:
        if beta == 0:
            return log_density1(z)
        elif beta == 1.:
            return log_density2(z)
        else:
            beta_ = torch.clamp(beta, min=1e-7, max=1-1e-7)
            return 1./alpha * torch.logsumexp(torch.cat([torch.log(1. - beta_) + alpha * log_density1(z), torch.log(beta_) + alpha * log_density2(z)], dim=1), dim=1, keepdim=True)
    
def annealing_path_power(log_density1, log_density2, alpha=0):
    return lambda beta: lambda z: log_power_mean(z, beta, alpha, log_density1, 
                                                    log_density2)

def get_annealing_path(log_density1, log_density2, path, alpha=0.):
    if path == 'geometric':
        return annealing_path_geometric(log_density1, log_density2)
    elif path == 'linear':
        return annealing_path_linear(log_density1, log_density2)
    elif path == 'power':
        return annealing_path_power(log_density1, log_density2, alpha)
    else:
        raise NotImplemented

def get_schedule(M, schedule):
    if schedule == 'geometric':
        if M == 1:
            return torch.tensor([1.], requires_grad=False)
        else:
            return torch.tensor(np.geomspace(.001, 1., M), requires_grad=False)
    elif schedule == 'linear':
        return torch.tensor(np.linspace(0., 1., M+1)[1:], requires_grad=False)
    elif schedule == 'sigmoid':
        scale = 0.3
        s = torch.sigmoid(torch.tensor( \
                    np.linspace(-1./scale, 1./scale, M + 1), requires_grad=False))
        return ((s - s[0]) / (s[-1] - s[0]))[1:]
    elif schedule == 'mcmc':
        return torch.ones(M, requires_grad=False)
    else:
        raise NotImplemented

        
def get_transition(input_dim, context_dim, M, transition, hidden_dim=4, 
                   step_size=0.5, update='fix', n_tune=1, partial_refresh=200, alpha=0.8, 
                   n_leapfrogs=1, r_hidden_dim=4, score=None):
    if transition in ['RWMH']:
        sampler = lambda z, x: torch.randn_like(z).to(z.device)
        log_density = lambda u, x: torch.distributions.Normal( \
                        loc=torch.tensor(0., device=u.device, dtype=torch.float32), \
                        scale=torch.tensor(1., device=u.device, dtype=torch.float32) \
                       ).log_prob(u).sum(dim=-1, keepdim=True)
        return MH(input_dim, context_dim, hidden_dim, sampler, log_density, 
                  symmetric=True, step_size=step_size, update=update, n_tune=n_tune) 
    elif transition == 'HMC':
        momentum_sampler = lambda shape: torch.randn(shape)
        momentum_log_density = lambda p: torch.distributions.Normal( \
                        loc=torch.tensor(0., device=p.device, dtype=torch.float32), \
                        scale=torch.tensor(1., device=p.device, dtype=torch.float32) \
                       ).log_prob(p).sum(dim=-1, keepdim=True)
        return HMC(input_dim, momentum_sampler, momentum_log_density, step_size, update=update, 
                                    n_tune=n_tune, partial_refresh=partial_refresh, alpha=alpha, 
                                   n_leapfrogs=n_leapfrogs)
        
    elif transition == 'MALA':
        return MALA(input_dim, step_size, update=update, n_tune=n_tune)
    elif transition == 'ULA':
        return ULA(input_dim, step_size, update=update, n_tune=n_tune)
    elif transition == 'ULA_S':
        return ULA_S(input_dim, step_size, update=update, n_tune=n_tune, score=score)
    else:
        raise NotImplemented

def get_f(alpha, log_u):
    if alpha == 0:
        return - log_u
    elif alpha == 1:
        return torch.exp(log_u) * log_u
    else:
        return (torch.exp(alpha * log_u) - 1 - alpha * (torch.exp(log_u) - 1)) / alpha / (alpha - 1.)