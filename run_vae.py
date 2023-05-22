import torch
import numpy as np
import os, pickle

from utils.experiments import get_parsed_args
from utils.data import make_dataloaders
from utils.experiments import get_modified_experiment, get_dirs
from models.vae import VAE, IWAE
from models.aux import load_checkpoint
from utils.gen_experiments import get_model_dirs, get_model_kwargs, Default_model_ARGS as model_args
from sampling.utils import get_sampler, estimate

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
        print(experiment)


        loaders, normalized = make_dataloaders(dataset=args.dataset, batch_sizes=[128] * 3, 
                                   ais_split=False, seed=args.seed, binarize=True, 
                                   dequantize=False)

        if args.target == 'vae50':
            args.latent_dim = 50
        elif args.target == 'vae100':
            args.latent_dim = 100

        model_args.testname = f'{args.testname}/models'
        model_args.model = 'VAE'
        model_args.latent_dim = args.latent_dim
        model_args.net = 'binary'
        model_args.likelihood = 'bernoulli'
        model_args.dataset = args.dataset
        model_args.binarize = True
        model_args.batch_sizes = [128, 128]
        model_args.epochs = 1000

        model_kwargs, code = get_model_kwargs(model_args)
        _, model_ckpt_dir, _ = get_model_dirs(model_args, code, make=False)

        model = VAE(**model_kwargs)

        model.cuda()
        model.eval()

        model = load_checkpoint(os.path.join(model_ckpt_dir, 'best.ckpt'), model.optimizers, False, model)
        print(f'Loaded checkpoint at epoch {model.current_epoch}')
        target_log_density = lambda z, x: model.log_joint_density(z, x, None, None)
        
        sampler, dirs = get_sampler(args, experiment, target_log_density, make_dirs=True)
        save_dir, _, _, results_dir = dirs
        
        done = os.path.join(save_dir, 'done')
        if not os.path.isfile(done) or args.redo:
            # if args.sampler == 'MCD':
            #     train(sampler)

            sampler.eval()

            if args.sampler in ['SD', 'Adaptive', 'SDSMC', 'AdaptiveSMC']:
                sampler.tune_beta = True
                with torch.no_grad():
                    for i, batch in enumerate(loaders[2]):
                        x, _ = batch
                        x = x.cuda()
                        z, log_w, transition_logs = sampler.sample(args.n_samples, x=x, update_step_size=False, log=False)
                        transition_logs['beta'] = torch.cat([torch.tensor([0.]).to(sampler.beta.device), sampler.beta], dim=0)
                        out['eval_transitions_tuning'].append(transition_logs)
                        break

                print('M = ', sampler.M)
                print(sampler.beta[-10:])

            sampler.tune_beta = False
            logs = []
            with torch.no_grad():
                for i, batch in enumerate(loaders[2]):
                    x, _ = batch
                    x = x.cuda()
                    z, log_w, transition_logs = sampler.sample(args.test_n_samples, x=x,  update_step_size=False, log=False)
                    transition_logs['beta'] = torch.cat([torch.tensor([0.]).to(sampler.beta.device), sampler.beta], dim=0)
                    # log_w = transition_logs['log_w'][:, -1]
                    # estimate(transition_logs, target_log_density, args.test_n_samples)
                    logs.append(transition_logs)
            out['eval_transitions'].append(logs)
        else:
            print('done', save_dir, seed)
            break

    if len(out['eval_transitions']) == len(seeds):
        with open(os.path.join(results_dir, 'all.pkl'), 'wb+') as f:
            pickle.dump(out, f)
        with open(done, 'w+') as f:
            f.write(':)')
        Ms = []
        logEws = []
        for transition_logs in out['eval_transitions']:
            Ms.append(transition_logs[0]['beta'].shape[0] - 1)
            logEws.append(transition_logs[0]['logZ'].mean().item())
        print(np.mean(Ms), np.mean(logEws))
    else:
        print('skipped')


