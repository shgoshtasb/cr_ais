import torch
import torch.nn as nn
import torchvision
import numpy as np

from sampling.annealing import get_transition, get_f
from sampling.ais import AIS
from utils.aux import repeat_data, get_ess, logselfnormalize, resample

class SMC(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='',
                 annealing_kwargs={'M':128, 'schedule':'geometric', 'path':'geometric', },
                 transition='RWMH', transition_kwargs={}, 
                 device=1, logger=None, name='SMCPlain', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, ckpt, annealing_kwargs,
                         transition, transition_kwargs, device, logger, name)
        for p in self.parameters():
            p.requires_grad_(False)

    def set_annealing(self, M, schedule, path, alpha=0., resampling=1, ratio=0.5):
        self.schedule = schedule
        self.beta_ = None
        self.path = path
        self.alpha = alpha
        self.resampling = resampling
        self.ratio = ratio
        self.M = M

    def sample(self, n_samples=1, x=None, update_step_size=False, log=False):
        z, log_probz = self.init_sample(n_samples, x)
        return self(n_samples, z, None, log_probz, update_step_size, log)

    def init_sample(self, n_samples=1, x=None):
        if x is not None and x.nelement() == 0:
            x = None
        self.set_context(x if x is None else repeat_data(x, n_samples))
        
        z_samples = n_samples if self.current_rx is None else self.current_rx.shape[0]
        z = self.current_proposal_sample(z_samples)
        log_probz = self.current_proposal_log_prob(z)
        return z, log_probz

    def update_resampling_thr(self, ess):
        return ess * self.ratio

    def get_expectation(self, h, log_w):
        norm_w = torch.exp(logselfnormalize(log_w))
        return (norm_w * h).sum(dim=0)

    def get_stats(self, z, log_w, logZ, logEw, beta):
        z_target_log_density = self.current_target_log_density(z)
        z_step_log_density = self.annealing_path(beta)(z)

        log_w_target = log_w - z_step_log_density + z_target_log_density
        log_w_step = log_w
        
        logEw_target = logEw - z_step_log_density + z_target_log_density
        logEw_step = logEw
        
        log_Zpi = logZ + torch.logsumexp(logEw_target, dim=0)
        log_Zq = logZ + torch.logsumexp(logEw_step, dim=0)
        log_Zratio = log_Zpi - log_Zq
        Zratio = torch.exp(log_Zratio)
        
        log_tilde_u = z_target_log_density - z_step_log_density
        log_u = log_tilde_u - log_Zratio
        
        f = get_f(self.alpha, log_u)

        # var_fstar = ((fstar - fstar.mean(dim=0))**2).mean()
        Vin = log_tilde_u if self.alpha == 0 else torch.exp(self.alpha * log_tilde_u)
        V = ((Vin - Vin.mean(dim=0))**2).mean()
        kl = - (log_u).mean()
        # f_mean = f.mean()
        
        ## importance weighted stats
        # var_fstar = (norm_w_step * (fstar - (norm_w_step * fstar).sum(dim=0))**2).sum()
        # V = (norm_w_step * (Vin - (norm_w_step * Vin).sum(dim=0))**2).sum()
        # V = self.var(Vin, log_w_step)
        # kl = - self.get_expectation(log_u, log_w_step)
        f_mean = self.get_expectation(f, log_w_step)
        # fstar_mean = self.get_expectation(fstar, log_w_step)
        
        if self.alpha == 0:
            ndlog_beta = V
        else:
            ndlog_beta = V / (self.alpha ** 2) / torch.exp(self.alpha * log_Zratio[0]) 

        return {'f':f_mean, 'ndlog_beta':ndlog_beta, 'kl':kl, 'log_Zratio':log_Zratio}

    def forward(self, n_samples, z, log_w, log_probz, update_step_size=False, log=False, transition_logs={}):
        if log_w is None:
            log_w = torch.zeros_like(log_probz)
        logZ = 0.
        logEw = logselfnormalize(log_w)
        thr = self.update_resampling_thr(n_samples)
            
        stats = self.get_stats(z, log_w, logZ, logEw, torch.zeros(1).to(z.device))
        transition_logs = self.update_transition_logs(z, log_w, stats=stats, log=log)
        
        last_log_density = log_probz
        for i in range(self.M):
            z, _, log_w_inc, logger_dict = self.transitions(z, log_w, 
                                             self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
            
            next_log_density = self.annealing_path(self.get_beta(i))(z)
            log_w = logselfnormalize(log_w) + log_w_inc - last_log_density + next_log_density
            logEw = logEw + log_w_inc - last_log_density + next_log_density
            last_log_density = next_log_density
            ess = get_ess(log_w)

            logZ_notresampled = logZ
            logEw_notresampled = logEw
            if (self.resampling > 0 and i > 0 and i % self.resampling == 0) or (self.resampling == -1 and ess < thr):
                ## Resample
                z, last_log_density = resample(z, log_w, next_log_density)
                logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
                thr = self.update_resampling_thr(get_ess(log_w))
                log_w = torch.zeros_like(log_probz)
                logEw = logselfnormalize(log_w)

            stats = self.get_stats(z, log_w, logZ_notresampled, logEw_notresampled, self.get_beta(i))
            transition_logs = self.update_transition_logs(z, log_w, transition_logs, logger_dict, stats=stats, log=log)
                
        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        
        logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
        transition_logs['logZ'] = logZ
        return z, log_w, transition_logs
        

class AdaptiveSMC(SMC):
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

    def set_annealing(self, path, alpha=0., resampling=1, max_step=1/8, min_step=1e-3, ratio=0.5):
        self.max_step = max_step
        self.min_step = min_step
        self.tune_beta = True
        self.beta_ = []
        self.path = path
        self.alpha = alpha
        self.resampling = resampling
        self.ada_ratio = ratio
        self.ratio = 0.9
        self.M = None
                        
    def get_cess(self, log_w, log_w_inc):
        log_nw = logselfnormalize(log_w)
        inv = torch.logsumexp(log_nw + 2 * log_w_inc, dim=0) - 2 * torch.logsumexp(log_nw + log_w_inc, dim=0)
        return torch.exp(-inv) * log_w.shape[0]
        
    def update_thr(self, cess, n_samples):
        return cess * self.ada_ratio
        ## For the absolute ess case instead of relative decrease
        # return n_samples * self.ratio
        
    def init_step(self, beta):
        if 1. - beta < self.min_step:
            return 1. - beta
        else:
            return min(self.max_step, 1. - beta)

    def tune_beta_step(self, n_samples, z, log_w, cess, update_step_size):
        # self.transitions.extend(get_transition(self.input_dim, self.context_dim, 1, self.transition, **self.transition_kwargs).to(self.device))

        last_beta = self.get_last_beta(z.device)
        step = self.init_step(last_beta)
        thr = self.update_thr(cess, n_samples)

        _, _, log_w_inc, _ = self.transitions(z, log_w, 
                        self.annealing_path(last_beta + step), update_step_size)
        cess = self.get_cess(log_w, log_w_inc)

        while cess < thr and step > self.min_step:
            step /= 2.
            _, _, log_w_inc, _ = self.transitions(z, log_w,
                        self.annealing_path(last_beta + step), last_beta + step, update_step_size)
            cess = self.get_cess(log_w, log_w_inc)
        self.beta_.append(last_beta + step)                    

    def forward(self, n_samples, z, log_w, log_probz, update_step_size=False, log=False, transition_logs={}):
        if log_w is None:
            log_w = torch.zeros_like(log_probz)
        logZ = 0.
        logEw = logselfnormalize(log_w)
        cess = n_samples
        thr = self.update_resampling_thr(n_samples)
            
        stats = self.get_stats(z, log_w, logZ, logEw, torch.zeros(1).to(z.device))
        transition_logs = self.update_transition_logs(z, log_w, stats=stats, log=log)
        
        last_log_density = log_probz
        i = 0
        while (self.tune_beta and (len(self.beta) == 0 or self.beta[-1] < 1)) or (not self.tune_beta and i < len(self.beta)):
            if self.tune_beta:
                self.tune_beta_step(n_samples, z, log_w, cess, update_step_size)
                
            z, _, log_w_inc, logger_dict = self.transitions(z, log_w,
                            self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
                            
            next_log_density = self.annealing_path(self.get_beta(i))(z)
            cess = self.get_cess(log_w, log_w_inc)
            log_w = logselfnormalize(log_w) + log_w_inc - last_log_density + next_log_density
            logEw = logEw + log_w_inc - last_log_density + next_log_density
            last_log_density = next_log_density
            ess = get_ess(log_w)

            logZ_notresampled = logZ
            logEw_notresampled = logEw
            if (self.resampling > 0 and i > 0 and i % self.resampling == 0) or (self.resampling == -1 and ess < thr):
                ## Resample
                z, last_log_density = resample(z, log_w, next_log_density)
                logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
                thr = self.update_resampling_thr(get_ess(log_w))
                log_w = torch.zeros_like(log_probz)
                logEw = logselfnormalize(log_w)

            stats = self.get_stats(z, log_w, logZ_notresampled, logEw_notresampled, self.get_beta(i))
            transition_logs = self.update_transition_logs(z, log_w, transition_logs, logger_dict, stats=stats, log=log)
            i += 1

        if self.tune_beta:
            self.beta_ = torch.stack(self.beta_)
            self.M = self.beta_.shape[0]

        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        
        logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
        transition_logs['logZ'] = logZ
        return z, log_w, transition_logs


class CRSMC(SMC):
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

    def set_annealing(self, path, alpha=0., resampling=1, ratio=0.5, dt=1., B=1., max_M=1024, tol=1e-7, kde=False):
        self.dt = dt
        self.B = B
        self.tol = tol
        self.max_M = max_M
        self.tune_beta = True
        self.beta_ = []
        self.path = path
        self.alpha = alpha
        self.beta_kde = kde
        self.max_step = 1/8
        self.beta_clip = True
        self.min_step = 5e-4
        self.separate = True
        self.resampling = resampling
        self.ratio = ratio
        self.M = None
                
    def tune_beta_step(self, ndlog_beta):
        # self.transitions.extend(get_transition(self.input_dim, self.context_dim, 1, self.transition, **self.transition_kwargs).to(self.device))
        
        last_beta = self.get_last_beta(ndlog_beta.device)
        beta = 1 - (1 - last_beta) * torch.exp(- self.dt * np.sqrt(self.B) / ndlog_beta)
        if self.beta_clip:
            beta = torch.clamp(beta, max=last_beta + self.max_step)
        beta = torch.clamp(beta, max=1.)
        self.beta_.append(beta)
        
    def forward(self, n_samples, z, log_w, log_probz, update_step_size=False, log=False, transition_logs={}):
        if log_w is None:
            log_w = torch.zeros_like(log_probz)
        logZ = 0.
        logEw = logselfnormalize(log_w)
        logZ_notresampled = logZ
        logEw_notresampled = logEw
        thr = self.update_resampling_thr(n_samples)
            
        stats = self.get_stats(z, log_w, logZ, logEw, torch.zeros(1).to(z.device))
        transition_logs = self.update_transition_logs(z, log_w, stats=stats, log=log)
        
        last_log_density = log_probz
        i = 0
        while (self.tune_beta and len(self.beta_) < self.max_M and stats['ndlog_beta'] > self.tol) or (not self.tune_beta and i < len(self.beta_)):
            if self.tune_beta:                   
                self.tune_beta_step(stats['ndlog_beta'])
                
            z, _, log_w_inc, logger_dict = self.transitions(z, log_w,
                            self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
                            
            next_log_density = self.annealing_path(self.get_beta(i))(z)
            log_w = logselfnormalize(log_w) + log_w_inc - last_log_density + next_log_density
            logEw = logEw + log_w_inc - last_log_density + next_log_density
            last_log_density = next_log_density
            ess = get_ess(log_w)

            logZ_notresampled = logZ
            logEw_notresampled = logEw
            if (self.resampling > 0 and i > 0 and i % self.resampling == 0) or (self.resampling == -1 and ess < thr):
                ## Resample
                z, last_log_density = resample(z, log_w, next_log_density)
                logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
                thr = self.update_resampling_thr(get_ess(log_w))
                log_w = torch.zeros_like(log_probz)
                logEw = logselfnormalize(log_w)

            stats = self.get_stats(z, log_w, logZ_notresampled, logEw_notresampled, self.get_beta(i))
            transition_logs = self.update_transition_logs(z, log_w, transition_logs, logger_dict, stats=stats, log=log)
            i += 1

        if self.tune_beta:
            if len(self.beta_) == 0 or self.beta_[-1] < 1.:
                self.beta_.append(torch.tensor(1.).to(self.device))
                # self.transitions.extend(get_transition(self.input_dim, self.context_dim, 1, self.transition, **self.transition_kwargs).to(self.device))

                z, log_w, _, logger_dict = self.transitions(z, log_w, 
                                self.annealing_path(self.get_beta(i)), self.get_beta(i), update_step_size)
                stats = self.get_stats(z, log_w, logZ_notresampled, logEw_notresampled, self.get_beta(i))
                transition_logs = self.update_transition_logs(z, log_w, 
                                                    transition_logs, logger_dict, stats=stats, log=log)
                
            self.beta_ = torch.stack(self.beta_)
            self.M = self.beta_.shape[0]

        for k in transition_logs.keys():
            transition_logs[k] = torch.cat(transition_logs[k], dim=-1)
        
        logZ = logZ + torch.logsumexp(logEw.view(n_samples, -1), dim=0)
        transition_logs['logZ'] = logZ
        return z, log_w, transition_logs
