import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from scipy.special import logsumexp
import numpy as np

from .aux import logselfnormalize
#xy axis limits for density heatmap
lim = 4
LIMS = np.array([[-lim, lim], [-lim, lim]])

def plot_particles(ax, z, log_w=None, legend=False, lims=LIMS, nb_point_per_dimension=100, cmap="coolwarm", alpha =0.5, label=''):
    print(label)
    if log_w is not None:
        min_w = torch.min(log_w, dim=0)[0]
        max_w = torch.max(log_w, dim=0)[0]
        scaled_w = torch.exp(log_w - max_w).cpu()
        marker_sizes = np.linspace(1, 100, 101)[(scaled_w * 100).int()]
        #scaled_log_w = ((w / torch.max(w)) * 100).cpu()
        #scaled_log_w = (torch.zeros(1024)).int().cpu()
        colors = cm.rainbow(np.linspace(0, 1, 101))
        heatmap = ax.scatter(z[:,0].cpu(), -z[:,1].cpu(), c=scaled_w, cmap=cmap, s=marker_sizes, label=label)#, rasterized=True)
    else:
        heatmap = ax.scatter(z[:,0].cpu(), -z[:,1].cpu(), marker='o', s=4, alpha=alpha, label=label)#, rasterized=True)
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])
    return heatmap

def plot_energy_heatmap(ax, log_density, x, lims=LIMS, nb_point_per_dimension=100, cmap="binary", contour=False):
    data_dim = 2
    xx, yy = np.meshgrid(np.linspace(lims[0][0], lims[0][1], nb_point_per_dimension), 
                         np.linspace(lims[1][0], lims[1][1], nb_point_per_dimension))
    z = torch.tensor(np.concatenate((xx.reshape(-1, 1), yy.reshape(-1, 1)), axis=1)).cuda()
    log_density = log_density(z.float(), x).reshape(nb_point_per_dimension, nb_point_per_dimension).detach().cpu()
    density = torch.exp(log_density)

    if cmap != 'blue':
        cm = cmap    
    else:
        colors = [(1, 1, 1), '#1f77b4']
        cm = LinearSegmentedColormap.from_list("Custom", colors, N=20)
        
    if contour:
        levels = np.linspace(log_density.min(),log_density.max(), 30)
        CS = ax.contour(xx, -yy, log_density, levels=levels)
        # ax.clabel(CS, inline=True, fontsize=10)
    else:
        ax.imshow(density, cmap=cm, aspect='auto', extent=lims.reshape(-1))
    
    ax.set_xlim(lims[0][0], lims[0][1])
    ax.set_ylim(lims[1][0], lims[1][1])

def plot_images(ax, x_gen):
    grid = torchvision.utils.make_grid(x_gen).detach().cpu()
    ax.imshow(grid.permute(1, 2, 0))
