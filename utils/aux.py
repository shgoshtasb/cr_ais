import torch

def secondsToStr(t):
    return "{:d}:{:02d}:{:02d}.{:03d}".format(int(t / 3600), int(t / 60) % 60, int(t) % 60, int(t * 1000) %1000)


def log_normal_density(z, mean, log_var):
    return torch.distributions.Normal(loc=mean, scale=torch.exp(0.5 * log_var)).log_prob(z).sum(dim=-1, keepdim=True)    

def repeat_data(x, n_samples):
    if len(x.shape) == 4:
        x = x.repeat(n_samples, 1, 1, 1)
    else:
        x = x.repeat(n_samples, 1)
    return x

def binary_crossentropy_logits_stable(x, y):
    return torch.clamp(x, 0) - x * y + torch.log(1 + torch.exp(-torch.abs(x)))


def binary_crossentropy_stable(s, y):
    x = torch.log(s + 1e-7) - torch.log(1 - s + 1e-7)
    return binary_crossentropy_logits_stable(x, y)

def get_bad_batch(zs, log_ws, loader, n_samples=1, batch_size=1):
    log_ws = log_ws.reshape(n_samples, -1)
    ll = log_ws.mean(dim=0)
    sort, bad_ind = torch.sort(ll)
    zs = zs.reshape(n_samples, -1, zs.shape[-1])[:,bad_ind]
    bad_x = torch.stack([loader.dataset.__getitem__(i)[0] for i in bad_ind[:batch_size]]).to(zs.device)
    return bad_x, zs[:,:batch_size], sort[:batch_size], bad_ind

def get_mean_std(x, numpy=True):
    mean = x.mean()
    std = torch.sqrt((x**2).mean() - x.mean()**2)
    if numpy:
        mean = mean.cpu().numpy()
        std = std.cpu().numpy()
    return mean, std

def logselfnormalize(log_w):
    scaled = log_w - torch.max(log_w, dim=0)[0]
    log_sum = torch.logsumexp(scaled, dim=0)
    return scaled - log_sum 
    
def get_ess(log_w):
    scaled_w = torch.exp(log_w - torch.max(log_w, dim=0)[0])
    return (scaled_w.sum(dim=0)**2)/(scaled_w**2).sum(dim=0)

def resample(z, log_w, log_density=None):
    normalized_w = torch.exp(log_w - torch.max(log_w, dim=0)[0])
    normalized_w = normalized_w / torch.sum(normalized_w, dim=0)
    dist = torch.distributions.Categorical(normalized_w.reshape(-1))
    ind = dist.sample(torch.Size([z.shape[0],]))
    if log_density is None:
        return z[ind]
    else:
        return z[ind], log_density[ind]
    
def freeze(*args):
    out_args = []
    for a in args:
        out_args.append(a.clone())
    return out_args
