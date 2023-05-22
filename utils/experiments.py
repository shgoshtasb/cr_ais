import itertools, os, shlex
from argparse import ArgumentParser
import numpy as np

def get_dirs(args, experiment, make=False, verbose=False):
    save_dir = '{}/{}/{}/{}'.format(args.testname, args.dataset, args.target, args.n_samples)
    save_dir = '{}/{}/{}'.format(save_dir, args.sampler, experiment['label'])
        
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    plot_dir = os.path.join(save_dir, 'plots')
    results_dir = os.path.join(save_dir, 'results')

    if make:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
    else:
        for dir_ in [save_dir, ckpt_dir, plot_dir, results_dir]:
            if not os.path.isdir(dir_):
                if verbose:
                    print(f'Wasn\'nt making and couldn\'t find {dir_}')
                return [None] * 4
    return save_dir, ckpt_dir, plot_dir, results_dir

def make_experiment(**kwargs):
    dict_ = {}
    for k in kwargs.keys():
        dict_[k] = kwargs[k]
    return dict_

def get_label(args):
    if args.sampler == 'Plain':
        label = f"{args.sampler[0]}{args.M}{args.schedule[0]}{args.path}{args.transition}{args.transition_update[0]}"
    elif args.sampler in ['Adaptive', 'CR']:
        label = f"{args.sampler[0]}{args.path}-{args.transition}{args.transition_update[0]}"
    elif args.sampler in ['SMC']:
        label = f"{args.sampler[0]}{args.M}{args.schedule[0]}{args.path}{args.transition}{args.transition_update[0]}-{args.resampling}"
    elif args.sampler in ['AdaptiveSMC', 'CRSMC']:
        label = f"{args.sampler[0]}{args.path}-{args.transition}{args.transition_update[0]}-{args.resampling}"        
    elif args.sampler in ['MCD']:
        label = f"{args.sampler[0]}{args.path}"
    return label

def get_transition_kwargs(args, kwargs):
    kwargs['transition'] = args.transition
    sym = [f'{args.transition}']
    kwargs['transition_kwargs'] = {}
    if args.transition in ['RWMH', 'MALA', 'ULA', 'HMC', 'ULA_S']:
        kwargs['transition_kwargs']['step_size'] = args.transition_step_size
        sym.append(f'{args.transition_step_size}')
    if args.transition == 'HMC':
        kwargs['transition_kwargs']['partial_refresh'] = args.hmc_partial_refresh
        kwargs['transition_kwargs']['alpha'] = args.hmc_alpha
        kwargs['transition_kwargs']['n_leapfrogs'] = args.hmc_n_leapfrogs
        sym.extend([f'{args.hmc_partial_refresh}', f'{args.hmc_alpha}{args.hmc_n_leapfrogs}'])
    if args.transition in ['RWMH', 'MALA', 'ULA', 'HMC']:
        kwargs['transition_kwargs']['update'] = args.transition_update
        if args.transition_update in ['tune', 'grad-std-tune']:
            kwargs['transition_kwargs']['n_tune'] = args.transition_n_tune
            update_sym = f'{args.transition_update[0]}{args.transition_n_tune}'
        else:
            update_sym = f'{args.transition_update[0]}'
    else:
        update_sym = ''
    return kwargs, '.'.join(sym) + update_sym

def get_annealing_kwargs(args, kwargs):
    kwargs['annealing_kwargs'] = {'path': args.path}
    if args.path == 'power':
        kwargs['annealing_kwargs']['alpha'] = args.annealing_alpha
        sym = f'p{args.annealing_alpha}'
    elif args.path == 'geometric':
        sym = f'g'
    if args.sampler not in ['CR', 'Adaptive', 'AdaptiveSMC','CRSMC']:
        kwargs['annealing_kwargs']['M'] = args.M
        sym += f'.{args.M}'
    if args.sampler not in ['CR', 'Adaptive', 'AdaptiveSMC','CRSMC', 'MCD']:
        kwargs['annealing_kwargs']['schedule'] = args.schedule
        sym += f'{args.schedule[0]}'
    if args.sampler in ['CR']:
        kwargs['annealing_kwargs']['dt'] = args.dt
        kwargs['annealing_kwargs']['tol'] = args.tol
        kwargs['annealing_kwargs']['max_M'] = args.max_M
        sym += f'.{args.dt}.{args.tol}.{args.max_M}'        
    elif args.sampler in ['Adaptive']:
        kwargs['annealing_kwargs']['max_step'] = args.max_step
        kwargs['annealing_kwargs']['min_step'] = args.min_step
        kwargs['annealing_kwargs']['ratio'] = args.ratio
        kwargs['annealing_kwargs']['conditional'] = args.conditional
        sym += f'.{args.min_step}.{args.max_step}'        
        sym += f'.{args.ratio}' + ('c' if args.conditional else '')
    elif args.sampler in ['SMC']:
        kwargs['annealing_kwargs']['resampling'] = args.resampling
        kwargs['annealing_kwargs']['ratio'] = args.ratio
        if args.resampling > 0:
            sym += f'.{args.resampling}'
        elif args.resampling == -1:
            sym += f'.a{args.ratio}'
    elif args.sampler in ['AdaptiveSMC']:
        kwargs['annealing_kwargs']['max_step'] = args.max_step
        kwargs['annealing_kwargs']['min_step'] = args.min_step
        kwargs['annealing_kwargs']['resampling'] = args.resampling
        kwargs['annealing_kwargs']['ratio'] = args.ratio
        sym += f'.{args.min_step}.{args.max_step}.{args.ratio}'
        if args.resampling > 0:
            sym += f'.{args.resampling}'
        elif args.resampling == -1:
            sym += f'.a'
    elif args.sampler in ['CRSMC']:
        kwargs['annealing_kwargs']['dt'] = args.dt
        kwargs['annealing_kwargs']['tol'] = args.tol
        kwargs['annealing_kwargs']['max_M'] = args.max_M
        kwargs['annealing_kwargs']['resampling'] = args.resampling
        kwargs['annealing_kwargs']['ratio'] = args.ratio
        sym += f'.{args.dt}.{args.tol}.{args.max_M}.{args.ratio}'
        if args.resampling > 0:
            sym += f'.{args.resampling}'
        elif args.resampling == -1:
            sym += f'.a'
    return kwargs, sym

def get_experiment(args):
    kwargs = {'sampler': args.sampler, 'M': args.M, 
              'input_dim':args.latent_dim, 'context_dim':args.context_dim}
    sym = {}
    
    kwargs, transition_sym = get_transition_kwargs(args, kwargs)
    sym['transition'] = transition_sym

    kwargs, annealing_sym = get_annealing_kwargs(args, kwargs)
    sym['rest'] = annealing_sym
    kwargs['label'] = '_'.join([sym[k] for k in sym.keys() if sym[k] != ''])
    return kwargs

class ARGS:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
class Default_ARGS():
    sampler='Plain'    
    testname="tests/"
    
    ## Target distribution
    target='U1'
    latent_dim=50
    dataset=None
    context_dim=0
    
    ## Sampling parameters
    seed = 1
    M=8
    n_samples=1024
    test_n_samples=1024
    schedule='geometric'
    path='geometric'
    device=1
    redo=False
    annealing_alpha=0.
    
    ## Training
    epochs=100
    learning_rate=3e-2
    
    ## Transitions
    transition='RWMH'
    transition_hidden_dim=4
    transition_step_size=0.5
    transition_update='fixed'
    transition_n_tune=10
    hmc_partial_refresh = 10
    hmc_alpha = 1.
    hmc_n_leapfrogs = 1
    
    ## CR-AIS
    tol=1e-7
    dt=1
    max_M=1024
    
    ## Adaptive
    min_step=1e-7
    max_step=1/128
    ratio=0.5
    conditional=False
    
    ## SMC
    resampling = -1
            
def get_modified_experiment(args, sampler, config = {}):
    experiment_kwargs = args.__dict__.copy()
    for updatekey in config.keys():
        experiment_kwargs[updatekey] = config[updatekey]
    args_ = ARGS(**experiment_kwargs)
    return args_, get_experiment(args_), get_label(args_)
    
def get_parsed_args(args_string=None):
    parser = ArgumentParser()
    args = Default_ARGS
    ## Sampler
    parser.add_argument("--sampler", default=args.sampler)

    ## Logging
    parser.add_argument("--testname", default=args.testname)

    ## Target distribution
    parser.add_argument("--target", default=args.target)
    parser.add_argument("--latent_dim", default=args.latent_dim, type=int)    
    parser.add_argument("--context_dim", default=args.context_dim, type=int)

    ## Dataset params
    parser.add_argument("--dataset", default=args.dataset, choices=[None, 'mnist', 'fashionmnist', 'cifar10', 'omniglot', 'celeba'])
    parser.add_argument("--binarize", action='store_true')
    parser.add_argument("--dequantize", action='store_true')
    
    ## Sampling parameters
    parser.add_argument("--seed", default=args.seed, type=int)
    parser.add_argument("--M", default=args.M, type=int)
    parser.add_argument("--n_samples", default=args.n_samples, type=int)
    parser.add_argument("--test_n_samples", default=args.test_n_samples, type=int)
    parser.add_argument("--schedule", default=args.schedule)
    parser.add_argument("--path", default=args.path)
    parser.add_argument("--device", default=args.device, type=int) 
    parser.add_argument("--redo", action='store_true')
    parser.add_argument("--annealing_alpha", type=float, default=args.annealing_alpha)
    
    ## Training
    parser.add_argument("--epochs", default=args.epochs, type=int) 
    parser.add_argument("--learning_rate", type=float, default=args.learning_rate)
    
    ## Transitions
    parser.add_argument("--transition", default=args.transition)
    parser.add_argument("--transition_update", default=args.transition_update)
    parser.add_argument("--transition_n_tune", default=args.transition_n_tune, type=int)
    parser.add_argument("--transition_hidden_dim", type=int, default=args.transition_hidden_dim)
    parser.add_argument("--transition_step_size", type=float, default=args.transition_step_size)
    parser.add_argument("--hmc_partial_refresh", type=int, default=args.hmc_partial_refresh)
    parser.add_argument("--hmc_alpha", type=float, default=args.hmc_alpha)
    parser.add_argument("--hmc_n_leapfrogs", type=int, default=args.hmc_n_leapfrogs)
    
    ## CR-AIS
    parser.add_argument("--tol", type=float, default=args.tol)
    parser.add_argument("--dt", type=float, default=args.dt)
    parser.add_argument("--max_M", type=int, default=args.max_M)
    
    ## Adaptive
    parser.add_argument("--min_step", type=float, default=args.min_step)
    parser.add_argument("--max_step", type=float, default=args.max_step)
    parser.add_argument("--ratio", type=float, default=args.ratio)
    parser.add_argument("--conditional", action='store_true')
    
    ## SMC
    parser.add_argument("--resampling", type=int, default=args.resampling)
    
    if args_string is not None:
        args = parser.parse_args(shlex.split(args_string))
    else:
        args = parser.parse_args()

    return args    

# List of terminal commands for the high dimensional experiments
def get_cmd(sampler, testname, target, latent_dim):
    transition = '--transition {} --transition_step_size {} --hmc_alpha 1. --hmc_partial_refresh 10 --hmc_n_leapfrogs 1 --transition_update fixed'
    annealing = '--path power --annealing_alpha {annealing_alpha} --max_M 2048 --tol 1e-3 --min_step=0.0000001'
    cmd = '--sampler {sampler} {transition} --n_samples 2048 --test_n_samples 4096 --testname {testname} --target {target} {annealing} --seed 1 --latent_dim {latent_dim}  {extra} --redo'
    
    annealing_alphas = [0., 0.25, 0.5, 0.75, 1., 2., -.5]
    max_steps = [1/4, 1/8, 1/16, 1./32, 1./64, 1./128, 1./256, 1./512]
    ratios = [0.9, 0.6, 0.7, 0.8]
    conditionals = ['', '--conditional']
    dts = {
        'UdNormal_128':[4096., 2048., 1024., 512., 256., 128., 64., 32., 16., 8., 4.],
        'UdMixture_128':[1024., 512., 256., 128., 64., 32., 16., 8., 4., 2., 1., 1./2],
        'UdLaplace_128':[4., 2., 1., 1./2, 1./4, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256],
        'UdStudentT_128':[1., 1./2, 1./4, 1./8, 1./16, 1./32, 1./64, 1./128, 1./256],
        'UdNormal_512':[4096., 2048., 1024., 512., 256., 128., 64., 32., 16., 8., 4.],
        'UdMixture_512':[2048., 1024., 512., 256., 128., 64., 32., 16., 8., 4., 2.],
        'UdLaplace_512':[8., 4., 2., 1., 1./2, 1./4, 1./8, 1./16, 1./32, 1./64],
        'UdStudentT_512':[8., 4., 2., 1., 1./2, 1./4, 1./8, 1./16, 1./32, 1./64],
    }
    Ms = [8, 16, 32, 64, 128, 256, 512, 1024]
    schedules = ['linear', 'sigmoid', 'geometric']

    cmds = []
    if sampler == 'CR':
        for annealing_alpha in annealing_alphas:
            for dt in dts[target]:
                cmd_ = cmd.format(sampler=sampler, 
                    transition=transition.format('HMC', 0.5), 
                    testname=testname, target=target, latent_dim=latent_dim,
                    annealing=annealing.format(annealing_alpha=annealing_alpha), 
                    extra=f'--dt {dt}')
                cmds.append(cmd_)
    if sampler == 'Adaptive':
        for annealing_alpha in annealing_alphas:
            for max_step in max_steps:
                for ratio in ratios:
                    for conditional in conditionals:
                        cmd_ = cmd.format(sampler=sampler, 
                            transition=transition.format('HMC', 0.5), 
                            testname=testname, target=target, latent_dim=latent_dim,  
                            annealing=annealing.format(annealing_alpha=annealing_alpha), 
                            extra=f'--max_step {max_step} --ratio {ratio} {conditional}')
                        cmds.append(cmd_)
            
    if sampler == 'Plain':
        for annealing_alpha in annealing_alphas:
            for M in Ms:
                for schedule in schedules:
                    cmd_ = cmd.format(sampler=sampler, 
                        transition=transition.format('HMC', 0.5), 
                        testname=testname, target=target, latent_dim=latent_dim,
                        annealing=annealing.format(annealing_alpha=annealing_alpha), 
                        extra=f'--M {M} --schedule {schedule}')
                    cmds.append(cmd_)
    if sampler == 'MCD':
        for annealing_alpha in annealing_alphas:
            for M in Ms:
                cmd_ = cmd.format(sampler=sampler, 
                    transition=transition.format('ULA_S', 0.01),
                    testname=testname, target=target, latent_dim=latent_dim,
                    annealing=annealing.format(annealing_alpha=annealing_alpha),
                    extra=f'--M {M}')
                cmds.append(cmd_)
    return cmds
