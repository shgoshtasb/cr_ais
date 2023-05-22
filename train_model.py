import torch 
import numpy as np
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os, pickle, time

from models.vae import VAE, IWAE
from models.aux import repeat_data, train, test
from utils.data import *
from models.aux import load_checkpoint, secondsToStr
from utils.gen_experiments import get_model_dirs, get_model_kwargs, get_model_parsed_args
    
if __name__ == '__main__':
    args = get_model_parsed_args()
    
    model_kwargs, code = get_model_kwargs(args)
    save_dir, ckpt_dir, log_dir = get_model_dirs(args, code, make=True)

    print(save_dir)
    #logger = SummaryWriter(log_dir=log_dir)
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    loaders, normalized = make_dataloaders(dataset=args.dataset, batch_sizes=args.batch_sizes, 
                               ais_split=False, seed=args.seed, binarize=args.binarize, 
                               dequantize=args.dequantize)

    if args.model == 'VAE':
        model = VAE(**model_kwargs)
    elif args.model == 'IWAE':
        model = IWAE(**model_kwargs)

    model.cuda()
    model.eval()

    start = time.time()
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')    
    if args.train_anyway or not os.path.isfile(last_ckpt):
        if os.path.isfile(os.path.join(save_dir, 'done')):
            os.remove(os.path.join(save_dir, 'done'))
        test(args, model, loaders[2], normalized, save_dir, n_samples=1, prefix='init')

        model.train()
        model, losses = train(args, model, 
                        loaders[0], loaders[1], normalized, save_dir, ckpt_dir, 
                        args.epochs, n_samples=args.n_samples)

    elif not os.path.isfile(os.path.join(save_dir, 'done')):
        model = load_checkpoint(last_ckpt, model.optimizers, False, model)
        print(f'Loaded checkpoint at epoch {model.current_epoch}')
        with open(os.path.join(save_dir, 'losses.pkl'), 'rb') as f:
            losses = pickle.load(f)
        for opt_idx, optimizer in enumerate(model.optimizers):
            for param_group in optimizer.param_groups:
                print(opt_idx, param_group['lr'])
                break
        model.train()
        model, losses = train(args, model, 
                        loaders[0], loaders[1], normalized, save_dir, ckpt_dir, 
                        args.epochs, losses, args.n_samples)
    else:
        best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')    
        model = load_checkpoint(best_ckpt, model.optimizers, False, model)
        print(f'Loaded checkpoint at epoch {model.current_epoch}')

    model.cuda()
    model.eval()
    test(args, model, loaders[2], normalized, save_dir, n_samples=args.n_samples, prefix='best')


    train_loss = losses['train']
    val_loss = losses['val']
    train_losses = losses['train_all']
    val_losses = losses['val_all']

    logs = {}
    for k in train_losses.keys():
        if k in logs.keys():
            logs[k][f'train'] = train_losses[k]
        else:
            logs[k] = {f'train':train_losses[k]}

    for k in val_losses.keys():
        if k in logs.keys():
            logs[k][f'val'] = val_losses[k]
        else:
            logs[k] = {f'val':val_losses[k]}

    c = len(logs.keys())
    fig, ax = plt.subplots(nrows=1, ncols=c + 1, figsize=(5*(c+1), 5))

    ax[0].plot(range(len(train_loss)), train_loss, label='train')
    ax[0].plot(range(len(val_loss)), val_loss, label='val')
    ax[0].set_title('loss')

    for i, k in enumerate(logs.keys()):
        for m in logs[k].keys():
            ax[i+1].plot(range(len(logs[k][m])), logs[k][m], label=m)
        ax[i+1].set_title(k)

    plt.savefig(os.path.join(save_dir, f'loss.png'))    
    end = time.time()
    print('Train time', secondsToStr(end - start))


