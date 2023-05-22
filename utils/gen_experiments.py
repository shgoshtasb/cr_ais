import itertools, os, shlex
from argparse import ArgumentParser

from .data import SHAPE

########################################################
## Generative model arguments and experiments
########################################################

class Default_model_ARGS():      #Wu     #iwae    #RD     #bigan
    # Logging 
    testname = 'generative_models'
    # Model arguments
    model = 'VAE'
    latent_dim = 10                   #1-2-5-10-20-50-100
    net = 'wu-wide'                   #wu-wide wu-small wu-shallow iwae dcgan conv
    deep = False
    likelihood = 'bernoulli'
    model_ckpt = None
    # Dataset arguments
    dataset = 'mnist'
    binarize = False
    dequantize = False
    # Training arguments
    batch_sizes = [128, 128]          #na     #20      #128    #128
    n_samples = 1                             #50
    learning_rate = [2e-4]            #1e-3 1e-4 1e-5 #2e-4    #2e-4
    epochs = 1000                     #1000   #3280   #1000?   #400    #dcgan 5 :))))
    #iwae likelihood bernoulli vae 86.76 1n_sample vae 86.35 iwae 84.78 50nsamples
    seed = 1
    device = 1
    train = False


def get_model_parsed_args(args_string=None):
    parser = ArgumentParser()
    ## Logging
    parser.add_argument("--testname", default=Default_model_ARGS.testname)

    ## Model
    parser.add_argument("--model", default=Default_model_ARGS.model)
    parser.add_argument("--likelihood", default=Default_model_ARGS.likelihood)
    parser.add_argument("--latent_dim", default=Default_model_ARGS.latent_dim, type=int)    
    parser.add_argument("--net", default=Default_model_ARGS.net)
    parser.add_argument("--deep", action='store_true')
    parser.add_argument("--model_ckpt", default=Default_model_ARGS.model_ckpt)

    ## Dataset
    parser.add_argument("--dataset", default=Default_model_ARGS.dataset)
    parser.add_argument("--binarize", action='store_true')
    parser.add_argument("--dequantize", action='store_true')
    
    ## Training
    parser.add_argument("--batch_sizes", default=Default_model_ARGS.batch_sizes, type=int, nargs='+')
    parser.add_argument("--n_samples", default=Default_model_ARGS.n_samples, type=int)
    parser.add_argument("--epochs", default=Default_model_ARGS.epochs, type=int)
    parser.add_argument("--learning_rate", default=Default_model_ARGS.learning_rate, type=float, nargs='+')
    parser.add_argument("--seed", default=Default_model_ARGS.seed, type=int)
    parser.add_argument("--device", default=Default_model_ARGS.device, type=int)
    parser.add_argument("--train_anyway", action='store_true')
    args = parser.parse_args()
    if args.model in ['VAE', 'IWAE']:
        args.learning_rate = [args.learning_rate[0]]
    elif args.model in ['GAN', 'BiGAN', 'AAE', 'WGANGP'] and len(args.learning_rate) < 2:
        args.learning_rate = [args.learning_rate[0]] * 2

    if args_string is not None:
        args = parser.parse_args(shlex.split(args_string))
    else:
        args = parser.parse_args()    
    return args

def get_model_default_kwargs(args = None):
    model_args = ARGS(**Default_model_ARGS.__dict__.copy())
    model_args.testname = 'generative_models'    
    model_args.model = args.target if args is not None else 'VAE'
    model_args.net = 'wu-wide'
    model_args.latent_dim = args.latent_dim if args is not None else 50
    model_args.learning_rate = [2e-4]
    if model_args.model in ['GAN', 'BiGAN', 'WGANGP']:
        model_args.learning_rate *= 2
    model_args.n_samples = 1
    if model_args.model in ['IWAE']:
        model_args.n_samples = 50
    model_args.epochs = 1000
    model_args.dataset = args.dataset if args is not None else 'mnist'
    model_args.binarize = args.binarize if args is not None else True
    model_args.dequantize = args.dequantize if args is not None else False
    model_args.device = args.device if args is not None else 2
    model_args.likelihood = 'bernoulli'
    model_args.train = False
    model_args.batch_sizes = [128, 128]
    model_args.seed = args.seed if args is not None else 1

    model_kwargs, code = get_model_kwargs(model_args)
    return model_args, model_kwargs, code
    
    
def get_model_kwargs(args):
    kwargs = {}
    kwargs['model'] = args.model
    attributes = ['latent_dim', 'net', 'deep',
            'dataset', 'learning_rate', 'device']
    if args.model == 'VAE':
        attributes.append('likelihood')
    elif args.model == 'IWAE':
        attributes.append('likelihood')
    #elif args.model == 'AAE':
    #    attributes.append('likelihood')
        
    model_kwargs = {}
    model_code = ''
    for attrib in attributes:
        model_kwargs[attrib] = getattr(args, attrib)
    model_kwargs['data_shape'] = SHAPE[args.dataset]
    
    model_code = f'{args.net[-1]}'
    if 'likelihood' in attributes:
        model_code += f'{args.likelihood[0]}'
    model_code += f'{args.latent_dim}'
    return model_kwargs, model_code

def get_model_dirs(args, code, make=False):
    dataset = args.dataset
    if args.binarize:
        dataset += 'b'
    elif args.dequantize:
        dataset += 'q'
    
    save_dir = '{}/{}/{}_{}_{}_{}_{}'.format(args.testname, 
            dataset, args.model, code, args.epochs,
            args.learning_rate, args.seed)
    if args.model in ['VAE', 'IWAE']:
        save_dir += '_{}'.format(args.n_samples)
    ckpt_dir = os.path.join(save_dir, 'checkpoints')
    log_dir = os.path.join(save_dir, 'log')
    if make:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    return save_dir, ckpt_dir, log_dir

