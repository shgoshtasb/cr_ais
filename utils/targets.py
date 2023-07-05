import torch
import numpy as np

from utils.mixtures import mixtures
from utils.data import Target

w1 = lambda z: torch.sin((2 * np.pi * z[:, 0]) / 4)
w2 = lambda z: 3 * torch.exp(-(((z[:, 0] - 1) / 0.6) ** 2) / 2)
w3 = lambda z: 3 * 1 / (1 + torch.exp(- ((z[:, 0] - 1) / 0.3)))

def bound_support(z, log_density, lim=8):
    return log_density - (((z > lim).to(torch.float32) + (z < lim).to(torch.float32)).sum(dim=-1, keepdim=True) > 0).to(torch.float32) * 100

class Normal(Target):
    def __init__(self, mean, std):
        super(Normal, self).__init__(mean.shape[1])
        self.mean = torch.nn.Parameter(mean, requires_grad=False)
        self.register_parameter('mean', self.mean)
        self.std = torch.nn.Parameter(std, requires_grad=False)
        self.register_parameter('std', self.std)
        self.distribution = torch.distributions.Normal(loc=self.mean, scale=self.std)
        
    @property
    def logZ(self):
        return .5 * self.data_dim * np.log(2 * np.pi) + torch.log(self.std).sum()
    
    def forward(self, z, x=None):
        return self.distribution.log_prob(z).sum(dim=1, keepdim=True) + self.logZ

class GaussianMixture(Target):
    def __init__(self, means, stds, pi=None):
        super(GaussianMixture, self).__init__(means.shape[1])
        self.means = torch.nn.Parameter(means, requires_grad=False)
        self.register_parameter('means', self.means)
        self.stds = torch.nn.Parameter(stds, requires_grad=False)
        self.register_parameter('stds', self.stds)
        if pi is None:
            pi = torch.ones(1, self.means.shape[2]).float()
        pi = pi / pi.sum()
        self.pi = torch.nn.Parameter(pi, requires_grad=False)
        self.register_parameter('pi', self.pi)
        
    @property 
    def logZ(self):
        return .5 * self.data_dim * np.log(2 * np.pi) + torch.logsumexp(torch.log(self.stds).sum(dim=1, keepdim=True), dim=2).sum()
        
    def forward(self, z, x=None):
        log_density = - 0.5 * (((z.reshape(z.shape + (1,)) - self.means)/self.stds)**2).sum(dim=1) -\
            0.5 * self.data_dim * np.log(2 * np.pi) -\
            torch.log(self.stds).sum(dim=1)
        return torch.logsumexp(log_density + torch.log(self.pi), dim=1, keepdim=True) +\
            self.logZ
    
class GaussianRing(GaussianMixture):
    def __init__(self, radius=3., std=0.1):
        rad = np.linspace(-1, 1, 9)[:-1] * np.pi
        means = torch.tensor(radius * np.stack([np.sin(rad), np.cos(rad)]), requires_grad=False, dtype=torch.float32).reshape(1, -1, 8)
        stds = torch.ones_like(means, requires_grad=False, dtype=torch.float32).reshape(1, -1, 8) * std
        super(GaussianRing, self).__init__(means, stds)
    
class EnergyBarrier(GaussianMixture):
    def __init__(self):
        pi = torch.Tensor([1., 1.]).reshape(1, 2).float()
        means = torch.tensor(np.ones((1, 2, 2)) * np.array([1., -2.]).reshape(1, 1, 2), requires_grad=False, dtype=torch.float32)
        stds = torch.ones((1, 2, 2), requires_grad=False, dtype=torch.float32) * np.array([.1, .2]).reshape(1, 1, 2)
        super(EnergyBarrier, self).__init__(means, stds, pi)
    
class NealNormal(Normal):
    def __init__(self):
        super(NealNormal, self).__init__(mean=torch.ones((1, 2), requires_grad=False).float(), 
                                         std=torch.ones((1, 2), requires_grad=False).float() * 0.1)
    
class U1(Target):
    def forward(self, z, x=None):
        return - ((((torch.sqrt(z[:, 0] ** 2 + z[:, 1] ** 2) - 2) / 0.4) ** 2) / 2 - torch.log(
    1e-15 + (torch.exp(-(((z[:, 0] - 2) / 0.6) ** 2) / 2) + torch.exp(-(((z[:, 0] + 2) / 0.6) ** 2) / 2)))).reshape(-1,1)
    
class U2(Target):
    def forward(self, z, x=None):
        return -((((z[:, 1] - w1(z)) / 0.4) ** 2) / 2).reshape(-1,1)
    
class U3(Target):
    def forward(self, z, x=None):
        return -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.35) ** 2) / 2) + torch.exp(-(((z[:, 1] - w1(z) + w2(z)) / 0.35) ** 2) / 2))).reshape(-1,1)
    
class U4(Target):
    def forward(self, z, x=None):
        return -(- torch.log(1e-15 + torch.exp(-(((z[:, 1] - w1(z)) / 0.4) ** 2) / 2) + torch.exp(
        -(((z[:, 1] - w1(z) + w3(z)) / 0.35) ** 2) / 2))).reshape(-1,1)

class Ud(GaussianMixture):
    def __init__(self, dim, std, max_n_components=128):
        means = []
        for i in range(max_n_components):
            means.append([(i//(2**(j)) % 2) if j < 10 else 0 for j in np.arange(dim)])
    
        means = torch.tensor(means)
        means = (means.reshape((1,) + means.shape).permute(0, 2, 1) - 0.5) * 10
        stds = torch.ones_like(means) * std
        super().__init__(means, stds)
        
class Distribution(Target):
    def __init__(self, dim):
        super(Distribution, self).__init__(dim)
        self.get_distribution()
        
    @property
    def logZ(self):
        return 0.
    
    def forward(self, z, x=None):
        return self.distribution.log_prob(z).sum(dim=1, keepdim=True) + self.logZ

class Laplace(Distribution):
    def get_distribution(self):
        self.mean = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
        self.register_parameter('mean', self.mean)
        self.register_parameter('std', self.std)

        self.distribution = torch.distributions.laplace.Laplace(self.mean, self.std)

class StudentT(Distribution):
    def get_distribution(self):
        self.df = torch.nn.Parameter(torch.tensor([3.0]).cuda(), requires_grad=False)
        self.register_parameter('df', self.df)
        self.distribution = torch.distributions.studentT.StudentT(self.df)
        
class Highdim(Target):
    def __init__(self, dim, distribution):
        super().__init__(dim)
        if distribution == 'Normal':
            self.target_log_density = Normal(mean = torch.ones((1, dim), requires_grad=False).float(),
                                             std = torch.ones((1, dim), requires_grad=False).float() * 0.1)
        elif distribution == 'Mixture':
            self.target_log_density = Ud(dim, std=1., max_n_components=8)
        elif distribution == 'StudentT':
            self.target_log_density = StudentT(dim)
        elif distribution == 'Laplace':
            self.target_log_density = Laplace(dim)
            
    @property
    def logZ(self):
        return torch.tensor([self.target_log_density.logZ]).reshape(-1,)
    
    def forward(self, z, x=None):
        return self.target_log_density(z, x)

mix = [{
        'means':torch.Tensor([[[ 2.6033, -0.1783],[ 0.8211, -1.1537]]]),
        'stds': torch.Tensor([[[0.5189, 0.6922],[0.6033, 0.9514]]]),
        'pi': torch.Tensor([[1.7381, 1.9599]])
    },{
        'means':torch.Tensor([[[-1.4348, -0.3754,  1.2525,  1.6261],
                             [-1.6012, -0.1817, -2.5013,  2.1189]]]),
        'stds': torch.Tensor([[[0.5296, 0.7707, 0.9555, 0.9492],
                             [0.6044, 0.7816, 1.0557, 0.6706]]]),
        'pi': torch.Tensor([[1.0541, 1.7398, 1.4888, 1.3785]])
    },{
    'means':torch.Tensor([[[ 0.6002,  1.8700,  0.6233, -1.5542,  0.9857, -0.6084, -2.3610, 2.3556],
                              [ 2.4665,  1.0264,  0.8344, -2.9071,  2.4241,  1.5608,  1.8788, -1.5171]]]),
         'stds': torch.Tensor([[[0.5096, 0.5464, 1.4801, 0.8559, 0.5436, 1.2275, 0.7444, 1.0689],
                              [1.4869, 0.6349, 1.4033, 0.8393, 0.5519, 0.5723, 1.2158, 1.2377]]]),
         'pi': torch.Tensor([[1.3583, 1.0846, 1.4591, 1.7067, 1.2923, 1.2409, 1.8130, 1.4815]])
    },{
        'means':torch.Tensor([[[-1.1953,  2.0248, -2.3643, -2.1961, -1.1098,  0.3471,  2.2692,
                               3.0468, -2.6092,  3.9793,  3.5525, -0.3497, -1.6294, -2.2386,
                              -1.8127, -1.9176],
                             [-2.8987, -1.1574, -0.7014,  3.4686, -2.0239,  1.1928, -1.4854,
                              -2.4378, -1.2179, -2.3622,  1.9503, -0.0893,  1.0405, -3.2335,
                              -3.3121,  1.4579]]]),
        'stds': torch.Tensor([[[1.1210, 0.7469, 1.4582, 1.0627, 0.9592, 1.1603, 0.6699, 1.2103,
                              0.8959, 0.8920, 0.5927, 0.6796, 0.7796, 1.4537, 1.2579, 1.1445],
                             [0.7969, 1.0471, 0.7628, 1.2789, 1.4794, 1.0173, 1.1333, 0.5733,
                              1.4220, 1.0076, 1.4669, 1.3258, 1.3203, 0.9253, 1.0933, 0.8275]]]),
        'pi': torch.Tensor([[1.6104, 1.7632, 1.1027, 1.1293, 1.3913, 1.6301, 1.6046, 1.4604, 1.4412,
                             1.8080, 1.6979, 1.2535, 1.0930, 1.4214, 1.3947, 1.7557]])
    }
]
# Energy functions
synthetic_targets = {
    "U0": Normal(mean=torch.zeros((1, 2), requires_grad=False).float(), 
                std=torch.ones((1, 2), requires_grad=False).float()),
    "U1": U1(),
    "U2": U2(),
    "U3": U3(),
    "U4": U4(),
    "U5": GaussianRing(),
    "U6": NealNormal(),
    #"U7": EnergyBarrier(),
}

# gaussian mixture distributions for 2d CR-AIS sampling experiments
for i, m in enumerate(mix):
    synthetic_targets[f'U{i+10}'] = GaussianMixture(m['means'], m['stds'], m['pi'])


def gaussian_mixture(z, mean):
    n_components = mean.shape[0]
    log_density = torch.logsumexp(-((z.reshape(z.shape + (1,)).repeat(1,1,n_components) - mean.reshape(mean.shape + (1,)).permute(2, 1, 0))**2).sum(dim=1)/2, dim=1)
    return log_density

# Wider gaussian mixture distributions for SMC
for i,m in enumerate(mixtures):
    mean = torch.Tensor(m['means']).reshape((1,) + m['means'].shape).permute(0, 2, 1) * 5
    std = torch.ones_like(mean) * 2
    pi = torch.ones(1,mean.shape[0])
    synthetic_targets[f'U{i+20}'] = GaussianMixture(mean, std)

for dim in [32, 128, 512]:
    for distribution in ['Normal', 'Mixture', 'Laplace', 'StudentT']:
        synthetic_targets[f'Ud{distribution}_{dim}'] = Highdim(dim, distribution)
