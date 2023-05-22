import torch
import numpy as np
import os, pickle, time

from sampling.ais import AIS
from models.flows import Gaussian

class MCD(AIS):
    def __init__(self, input_dim, context_dim, target_log_density, ckpt='',
                 annealing_kwargs={'M':128, 'schedule':'geometric', 'path':'geometric', },
                 transition='ULA_S', transition_kwargs={}, 
                 device=1, logger=None, name='MCD', **kwargs):
        super().__init__(input_dim, context_dim, target_log_density, ckpt,
                         annealing_kwargs, transition, transition_kwargs,
                         device, logger, name, **kwargs)

        # for checkpoint storing
        self.current_epoch = 0
        self.best_epoch = 0.
        self.best_loss = torch.tensor([np.inf]).to(self.device)
        
    def set_annealing(self, M, path, alpha, epochs=100, learning_rate=0.01, batch_size=128):
        super().set_annealing(None, M, path, alpha)
        self.score = Score(self.input_dim, self.M, dh=512, dt=16, k=3)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        if self.ckpt is not None:
            self.best_ckpt = os.path.join(self.ckpt, 'best.ckpt')
            self.last_ckpt = os.path.join(self.ckpt, '{}.ckpt')
            self.losses = os.path.join(self.ckpt, 'losses.pkl')
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_loss = np.inf
        self.register_parameter('beta_p', torch.nn.Parameter(torch.zeros(self.M, dtype=torch.float32)))
        
    @property
    def beta(self):
        return torch.cumsum(torch.softmax(self.beta_p, dim=0), dim=0)

    def set_transition(self, transition, **kwargs):
        kwargs['score'] = self.score
        super().set_transition(transition, **kwargs)
        
    def set_proposal(self):
        self.proposal = Gaussian(self.input_dim, trainable=True)

        
class Score(torch.nn.Module):
    def __init__(self, dim, M, dh, dt, k):
        super().__init__()
        self.t_map = torch.nn.Embedding(M + 1, dt)
        self.x_linear = torch.nn.Linear(dim, dh)
        self.modules__ = torch.nn.ModuleList([ResidualBlock(dh, dt) for i in range(k)])
        self.linear = torch.nn.Linear(dh, 1)
        self.M = M
        
        # last layer initialization
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x, t):
        t_map = self.t_map((t.detach() * self.M).long().detach().reshape(-1,).repeat(x.shape[0],))
        x = self.x_linear(x)
        for m in self.modules__[:-1]:
            x = m(x, t_map)
        return self.linear(x)
        
class Swish(torch.nn.Module):
    def __init__(self, dim):
        super(Swish, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones((1, dim), requires_grad=True))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)
                                               

class ResidualBlock(torch.nn.Module):
    def __init__(self, dh, dt):
        super().__init__()
        self.net1 = torch.nn.Sequential(torch.nn.LayerNorm(dh), Swish(dh), 
                                        torch.nn.Linear(dh, 2 * dh))
        self.linear = torch.nn.Linear(dt, 2 * dh)
        self.net2 = torch.nn.Sequential(Swish(2 * dh), torch.nn.Linear(2 * dh, dh), Swish(dh))
        
    def forward(self, x, t):
        return self.net2(self.net1(x) + self.linear(t)) + x
                                        

def load_ckpt(model, path):
    print("Loading checkpoint from: {}".format(path))
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint["state_dict"])
    model.current_epoch = checkpoint["current_epoch"]
    model.best_epoch = checkpoint["best_epoch"]
    model.best_loss = checkpoint["best_loss"]

    return model

def save_ckpt(model, path):
    save_dict = {"state_dict": model.state_dict(),
                "current_epoch": model.current_epoch,
                "best_epoch": model.best_epoch,
                "best_loss": model.best_loss}
    torch.save(save_dict, path)

def train(model):    
    optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)    
    losses = {'train':[], 'val':[]}

    start = time.time()
    while model.current_epoch < model.epochs:
        model.train()
        z,log_w, transition_logs = model.sample(model.batch_size, log=False)
        # loss = -transition_logs['logZ']
        loss = -log_w.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses['train'].append(loss.item())

        model.eval()
        z,log_w, transition_logs = model.sample(model.batch_size, log=False)
        loss = - log_w.mean() # -transition_logs['logZ']
        losses['val'].append(loss.item())
        
        if loss < model.best_loss:
            model.best_loss = loss.item()
            model.best_epoch = model.current_epoch
            save_ckpt(model, model.best_ckpt)
        if model.current_epoch - model.best_epoch >= 250:
            losses['early_stop'] = True
            break
        if model.current_epoch % int(model.epochs / 10) == 0:
            end = time.time()
            log = f'{secondsToStr(end - start)} Epoch {model.current_epoch}:'
            log += f' Train {losses["train"][-1]}, Val {losses["val"][-1]}'
            log += f' Best {model.best_loss} at {model.best_epoch}'
            print(log)
            start = end
        if model.current_epoch in [30, 50, 99]:
            save_ckpt(model, model.last_ckpt.format(model.current_epoch))
            with open(model.losses, 'wb+') as f:
                pickle.dump(losses, f)

        model.current_epoch += 1

    save_ckpt(model, model.last_ckpt.format(model.current_epoch))
    with open(model.losses, 'wb+') as f:
        pickle.dump(losses, f)
    _ = load_ckpt(model, model.best_ckpt)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return losses
    
def secondsToStr(t):
    return "{:d}:{:02d}:{:02d}.{:03d}".format(int(t / 3600), int(t / 60) % 60, int(t) % 60, int(t * 1000) %1000)

