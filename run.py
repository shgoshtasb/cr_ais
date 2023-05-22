#################################################
# Run sampling algorithm
#################################################

import torch
import numpy as np
import os, pickle
from copy import copy

from utils.experiments import get_parsed_args
from utils.targets import synthetic_targets
from utils.data import Pima, Sonar
from utils.experiments import get_modified_experiment, get_dirs
from sampling.utils import get_sampler, eeight_stats, stimate
from sampling.mcd import train

seeds = [1, 2, 3, 4, 5]
if __name__ == '__main__':
    args = get_parsed_args()
    out = {'eval_transitions':[], 'eval_transitions_tuning':[]}
    for seed in seeds:
        args.seed = seed
        args, experiment, label = get_modified_experiment(args, args.sampler, config={
            'transition_step_size':args.transition_step_size,
            'transition_update':args.transition_update,
            'transition_n_tune':args.transition_n_tune,
            'hmc_partial_refresh': args.hmc_partial_refresh,
            'hmc_alpha': args.hmc_alpha,
            'hmc_n_leapfrogs': args.hmc_n_leapfrogs,
        })

        if args.target in synthetic_targets.keys():
            target_log_density = synthetic_targets[args.target]
        elif args.target == 'pima':
            target_log_density = Pima()
        elif args.target == 'sonar':
            target_log_density = Sonar()
        sampler, dirs = get_sampler(args, experiment, target_log_density, make_dirs=True)

        save_dir, ckpt_dir, plot_dir, results_dir = dirs        
        print(save_dir)

        done = os.path.join(save_dir, 'done')
        if not os.path.isfile(done) or args.redo:
            if args.sampler == 'MCD':
                train(sampler)
                
            sampler.eval()
            
            if args.sampler in ['CR', 'Adaptive', 'CRSMC', 'AdaptiveSMC']:
                sampler.tune_beta = True
                z, log_w, transition_logs = sampler.sample(args.n_samples, None, update_step_size=True, log=False)
                transition_logs['beta'] = torch.cat([torch.tensor([0.]).to(sampler.beta.device), sampler.beta], dim=0)
                out['eval_transitions_tuning'].append(transition_logs)

                print('M = ', sampler.M)
                print(sampler.beta[-10:])

            sampler.tune_beta = False
            z, log_w, transition_logs = sampler.sample(args.test_n_samples, None, update_step_size=False, log=False)
            transition_logs['beta'] = torch.cat([torch.tensor([0.]).to(sampler.beta.device), sampler.beta], dim=0)
            log_w = transition_logs['log_w'][:, -1]
            estimate(transition_logs, target_log_density, args.test_n_samples)
            out['eval_transitions'].append(transition_logs)
        else:
            print('done', save_dir, seed)
            break

    Ms = []
    logEws = []
    for transition_logs in out['eval_transitions']:
        Ms.append(transition_logs['beta'].shape[0] - 1)
        logEws.append(transition_logs['logZ'].item())
    print(np.mean(Ms), np.mean(logEws))
                
    if len(out['eval_transitions']) == len(seeds):
        with open(os.path.join(results_dir, 'all.pkl'), 'wb+') as f:
            pickle.dump(out, f)
        with open(done, 'w+') as f:
            f.write(':)')
    else:
        print('skipped')
