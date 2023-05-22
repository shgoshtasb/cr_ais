import os, pickle, time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

from utils.aux import secondsToStr, get_mean_std, get_ess
from sampling.ais import Plain, CRAIS, AdaptiveAIS
from sampling.smc import SMC, AdaptiveSMC, CRSMC
from sampling.mcd import MCD
from utils.experiments import get_dirs

def tune_sampler(args, sampler, log_density, data_loader, experiment, losses, ckpt_dir):
    n_samples = args.n_samples
    sampler.eval()
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            if len(data_loader) == 1 or batch_idx % int(max(len(data_loader) / 5, 1)) == 0:
                end = time.time()
                print(f'Tune {batch_idx * batch_size} / {len(data_loader) * batch_size}', secondsToStr(end - start))
                start = end

            z, log_w, reinforce_log_prob, transition_logs = sampler(n_samples, x=x, 
                                                        update_step_size=True, log=False)
    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    save_sampler_ckpt(last_ckpt, [], False, sampler)
    save_sampler_ckpt(best_ckpt, [], False, sampler)
    return sampler, losses

def estimate(transition_logs, target_log_density, n_samples, prefix='', just_log=True):
    log_ws = transition_logs['log_w'].reshape(n_samples, -1)
    logZ = transition_logs['logZ'].detach().cpu().numpy()[0]
    Z = np.exp(logZ)
    exp_logZ = torch.exp(target_log_density.logZ) if target_log_density.logZ is not None else None
    logZs = None if exp_logZ is None else "{:.3f}".format(target_log_density.logZ.item())
    Zs = None if exp_logZ is None else "{:.3f}".format(exp_logZ.item())

    ESS = get_ess(log_ws)
    print("log Z: {}, log E w: {:.3f}".format(logZs, logZ))
    print("Z    : {}, E w    : {:.3f}".format(Zs, Z))
    print("ESS  : ({:.1f} +- {:.1f})/{}".format(*get_mean_std(ESS), n_samples))
    if not just_log:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
        ax.hist(log_ws[:,-1].cpu().numpy(), bins=100)
        ax.set_title(f'log w {prefix}')            
    
sampler_dict = {
    'Plain': Plain, 
    'CR': CRAIS,
    'Adaptive': AdaptiveAIS,
    'MCD': MCD,
    'SMC': SMC,
    'AdaptiveSMC': AdaptiveSMC,
    'CRSMC': CRSMC,
}

def get_sampler(args, experiment, target_log_density, make_dirs=False):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dirs = get_dirs(args, experiment, make=make_dirs)
    save_dir, ckpt_dir, plot_dir, results_dir = dirs        
    #with open(os.path.join(save_dir, 'args.pkl'), 'wb+') as f:
        #pickle.dump(args.__dict__.__str__(), f)

    sampler = sampler_dict[args.sampler](target_log_density=target_log_density, ckpt=ckpt_dir, **experiment)
    device = 'cuda' if args.device > 0 else 'cpu'
    sampler.to(device)
    return sampler, dirs
                
