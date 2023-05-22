import torch
import os, pickle
import numpy as np
from time import time

from utils.aux import secondsToStr, repeat_data, log_normal_density, get_bad_batch, get_mean_std
from .vae import VAE, IWAE

def load_checkpoint(path, optimizer, reset_optimizer, model, replace=[]):
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)
    
    for f, r in replace:
        for key in list(checkpoint['state_dict'].keys()):
            if f in key:
                checkpoint['state_dict'][key.replace(f, r, 1)] = checkpoint['state_dict'][key]
                del checkpoint['state_dict'][key]

    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and not reset_optimizer:
        model.current_epoch = checkpoint["current_epoch"]
        model.best_epoch = checkpoint["best_epoch"]
        model.best_loss = checkpoint["best_loss"]
        for opt_idx, opt in enumerate(optimizer):
            opt.load_state_dict(checkpoint[f'optimizer_{opt_idx}'])
    print(f'Loaded checkpoint at epoch {model.current_epoch}')
    return model

def save_checkpoint(checkpoint_path, optimizer, save_optimizer_state, model):
    save_dict = {"state_dict": model.state_dict(),
                "current_epoch": model.current_epoch,
                "best_epoch": model.best_epoch,
                "best_loss": model.best_loss,
                "log_var": model.log_var}
    if save_optimizer_state:
        for opt_idx, opt in enumerate(optimizer):
            save_dict[f'optimizer_{opt_idx}'] = opt.state_dict()
            
    torch.save(save_dict, checkpoint_path)

def get_model(args, model_kwargs, ckpt, device):
    if args.model == 'VAE':
        model = VAE(**model_kwargs)
    elif args.model == 'IWAE':
        model = IWAE(**model_kwargs)
        
    if os.path.isfile(ckpt):
        if device > 1:
            replace = [('.net.', '.module.net.')]
        else:
            replace = []
        model = load_checkpoint(ckpt, model.optimizers, False, model, replace)
        model.cuda()
        model.eval()

        for p in model.parameters():
            p.requires_grad = False

        return model

    else:
        print('Untrained model', ckpt)
        return None
    
    
def train(args, model, train_loader, val_loader, normalized, save_dir, ckpt_dir, epochs, losses={}, n_samples=1):
    start = time()
    
    if len(losses.keys()) == 0:
        losses['train'] = []
        losses['train_all'] = {}
        losses['val'] = []
        losses['val_all'] = {}

    best_ckpt = os.path.join(ckpt_dir, 'best.ckpt')
    last_ckpt = os.path.join(ckpt_dir, 'last.ckpt')
    min_epoch = model.best_epoch
    min_loss = model.best_loss
    epoch = model.current_epoch
    
    if epoch < epochs:
        while True:
            model.train()
            torch.set_grad_enabled(True)
            loss, loss_dict = model.train_epoch(train_loader, n_samples)

            losses['train'].append(loss)
            for k in loss_dict.keys():
                if k in losses['train_all'].keys():
                    losses['train_all'][k].append(loss_dict[k])
                else: 
                    losses['train_all'][k] = [loss_dict[k]]

            model.eval()
            with torch.no_grad():
                val_loss, val_dict = model.validation_epoch(val_loader, n_samples)

                losses['val'].append(val_loss)
                for k in val_dict.keys():
                    if k in losses['val_all'].keys():
                        losses['val_all'][k].append(val_dict[k])
                    else:
                        losses['val_all'][k] = [val_dict[k]]

                if val_loss < min_loss:
                    min_loss = val_loss
                    min_epoch = epoch
                    #torch.save(model.state_dict(), best_ckpt)
                    save_checkpoint(best_ckpt, model.optimizers, True, model)
                    if epoch - min_epoch >= 250:
                        print('early break')
                        break
            if epoch % (epochs / 10) == 0:
                end = time()
                print('Epoch', epoch, model.current_epoch, min_epoch, min_loss, secondsToStr(end-start))
                start = end
                log1 = f'Epoch {epoch} ' + \
                    ' '.join(['opt_{}: {:.3f}'.format(opt_idx, loss[opt_idx]) for opt_idx in range(len(loss))]) + \
                    ' val: {:.3f}'.format(val_loss)
                log2 = f'  Train ' + \
                    ' '.join(['{}: {:.3f}'.format(k, loss_dict[k]) for k in loss_dict.keys()]) + \
                    ' Val ' + ' '.join(['{}: {:.3f}'.format(k, val_dict[k]) for k in val_dict.keys()])            
                print(log1)
                print(log2)
                test(args, model, val_loader, normalized, save_dir, n_samples)
                save_checkpoint(last_ckpt, model.optimizers, True, model)
                with open(os.path.join(save_dir, 'losses.pkl'), 'wb+') as f:
                    pickle.dump(losses, f)
                model.best_epoch = min_epoch
                model.best_loss = min_loss

            if model.current_epoch in [100, 300, 500]:
                test(args, model, val_loader, normalized, save_dir, n_samples, prefix=f'{model.current_epoch}')
                epoch_ckpt = os.path.join(ckpt_dir, f'{model.current_epoch}.ckpt')
                save_checkpoint(epoch_ckpt, model.optimizers, True, model)
                
            epoch += 1
            if epoch == epochs:
                break
        test(args, model, val_loader, normalized, save_dir, n_samples, prefix='last')
        #torch.save(model.state_dict(), last_ckpt)
        save_checkpoint(last_ckpt, model.optimizers, True, model)
        with open(os.path.join(save_dir, 'losses.pkl'), 'wb+') as f:
            pickle.dump(losses, f)
        model.best_epoch = min_epoch
        model.best_loss = min_loss
        model = load_checkpoint(best_ckpt, model.optimizers, False, model)
        print(f'Loaded checkpoint at epoch {min_epoch}')
        with open(os.path.join(save_dir, 'done'), 'w+') as f:
            f.write(':)')
    return model, losses
    

def test(args, model, test_loader, normalized, save_dir, n_samples=1, prefix=None, save=False, save_img=False):
    model.eval()
    with torch.no_grad():
        rows = 8
        z = torch.randn(rows * rows, args.latent_dim).cuda().float()
        x, img = model.save_gen(z, None, normalized, rows, save_dir, prefix, save)

        log_ws = []
        zs = []
        nll = []
        iw = []
        for batch_idx, batch in enumerate(test_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            mean, log_var = model.encode(x)
            mean = mean.repeat(n_samples, 1)
            log_var = log_var.repeat(n_samples, 1)
            z, log_q = model.reparameterize(mean, log_var)
            x = repeat_data(x, n_samples)
            x_recon, x_log_var = model.decoder_with_var(z)
            if x_log_var is None:
                x_log_var = model.log_var
                
            log_joint_density = model.log_joint_density(z, x, x_recon, x_log_var)
            log_w = log_joint_density - log_q
            log_ws.append(log_w)
            zs.append(z)
            log_w = log_w.reshape(n_samples, -1)
            iw.append(torch.logsumexp(log_w, dim=0) - np.log(n_samples))
            nll.append(log_w.mean(dim=0))
        log_ws = torch.cat(log_ws, dim=0)
        zs = torch.cat(zs, dim=0)    
        iw = -torch.cat(iw, dim=0)
        nll = -torch.cat(nll, dim=0)
        scaled_w = torch.exp(log_ws.reshape(n_samples, -1) - torch.max(log_ws.reshape(n_samples, -1), dim=0)[0].reshape(1, -1))
        ESS = (scaled_w.sum(dim=0).reshape(1, -1)**2)/(scaled_w**2).sum(dim=0)
        print("NLL E log w: {:.3f}, std log w: {:.3f}".format(*get_mean_std(nll)))
        print("IW log E w: {:.3f}, std log w: {:.3f}".format(*get_mean_std(iw)))
        print("ESS  : ({:.1f} +- {:.1f})/{}".format(*get_mean_std(ESS), n_samples))

        recon_images(args, model, test_loader, normalized, save_dir, prefix, save, save_img)

        bad_x, bad_z, bad_ll, bad_ind = get_bad_batch(zs, log_ws, test_loader, n_samples, 16)
        print('bad nll', bad_ll.mean().cpu().numpy())
        model.save_gen(None, bad_x, None, 16, save_dir='recon_plots', 
                   prefix=f'{prefix}_data_bad', save=save_img)
        model.save_gen(bad_z[:8].reshape(-1, bad_z.shape[-1]), None, None, 16, save_dir='recon_plots', 
                   prefix=f'{prefix}_encoder_bad', save=save_img)
        


def recon_images(args, model, test_loader, normalized, save_dir, prefix=None, save=False, save_img=False):
    model.eval()
    with torch.no_grad():
        rx = []
        zs = []
        N = 0
        n_samples = 8
        rows = 16
        
        for batch_idx, batch in enumerate(test_loader):
            x, _ = batch
            x = x.cuda()
            batch_size = x.shape[0]
            if N < rows:
                rx.append(x[:rows])
            mean, log_var = model.encode(x)
            mean = mean.repeat(n_samples, 1)
            log_var = log_var.repeat(n_samples, 1)
            z, log_q = model.reparameterize(mean, log_var)
            if N < rows:
                zs.append(z.reshape(n_samples, batch_size, -1)[:,:16])
                N += rx[-1].shape[0]
            if N >= rows:
                break
        rx = torch.cat(rx, dim=0)
        zs = torch.cat(zs, dim=1).reshape(n_samples * rows, -1)
                
        model.save_gen(None, rx, None, rows, save_dir='recon_plots', 
                   prefix=f'{prefix}_data', save=save_img)
        model.save_gen(zs, None, None, rows, save_dir='recon_plots', 
                   prefix=f'{prefix}_encoder', save=save_img)
