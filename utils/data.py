import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
from random import shuffle


from utils.aux import binary_crossentropy_logits_stable

class Target(torch.nn.Module):
    def __init__(self, data_dim=2):
        super(Target, self).__init__()
        self.data_dim = data_dim
        
    @property
    def logZ(self):
        pass
    
    @property
    def get_all(self):
        return self, self.logZ, self.data_dim
    
    def forward(self, z, x=None):
        pass
    
def whiten(x):
    x = (x - x.mean(dim=1, keepdim=True))
    x = x / torch.sqrt((x**2).mean(dim=1, keepdim=True))
    return x    
    
class LogisticRegression(Target):
    def __init__(self, dim, x, y, sigma=1.):
        super().__init__(dim)
        self.x = whiten(x)
        self.y = y
        self.sigma = sigma
        
    def forward(self, z, x=None):
        base = torch.distributions.Normal(loc=torch.zeros(self.data_dim).to(z.device), scale=self.sigma * torch.ones(self.data_dim).to(z.device))
        prior = base.log_prob(z).sum(dim=1, keepdim=True)
        likelihood = binary_crossentropy_logits_stable(
                    torch.matmul(z, self.x.to(z.device)), self.y.to(z.device)).sum(dim=1, keepdim=True)
        return prior + likelihood
    
    @property
    def logZ(self):
        return None
            
class Pima(LogisticRegression):
    def __init__(self):
        df = pd.read_csv('pima-indians-diabetes.csv')
        x = torch.tensor(df.values[:,:-1], requires_grad=False).float().permute(1,0)
        y = torch.tensor(df.values[:,-1], requires_grad=False).float().reshape(1, -1)
        super().__init__(8, x, y, 5.)
        
    
class Sonar(LogisticRegression):
    def __init__(self):
        df = pd.read_csv('sonar.all-data.csv')

        x = torch.tensor(df.values[:,:-1].astype(np.float32),requires_grad=False).float().permute(1,0)
        y = torch.tensor((df.values[:,-1] == 'M'), requires_grad=False).float().reshape(1, -1)
        super().__init__(60, x, y, 5.)


# Wu et al. https://arxiv.org/abs/1611.04273
# For consistency with prior work on evaluating decoder-based models 
# 1. dequantized the data as in Uria et al. (2013), by
# 2. use 50000 samples form training split for training generative models and remaining 10000 
# for validation and training AIS
# 3. binarized MNIST with a Bernoulli observation likelihood possible (Salakhutdinov & Murray, 2008)
#
# model comparison for fashionmnist, cifar10, omniglot, celeba datasets 


SHAPE = {
    'mnist': torch.tensor((1, 28, 28)),
    'fashionmnist': torch.tensor((1, 28, 28)),
    'omniglot': torch.tensor((1, 105, 105)),
    'cifar10': torch.tensor((3, 32, 32)),
    'celeba': torch.tensor((3, 64, 64)),
    None: torch.tensor([])
}

VALRATIO = {
    'mnist': 1./6,
    'fashionmnist': 1./6,
    'omniglot': 1./8,
    'cifar10': 1./5,
    'celeba': 1.,
}

AISRATIO = {
    'mnist': 1./5,
    'fashionmnist': 1./5,
    'omniglot': 1./5,
    'cifar10': 1./4,
    'celeba': 1./6,
}

    
class NoneDataset(Dataset):
    def __init__(self):
        super(NoneDataset, self).__init__()
        
    def __len__(self):
        return 1
    
    def __getitem__(self, item):
        return torch.tensor([]), torch.tensor([])
        
def make_dataloaders(dataset, batch_sizes, ais_split=True, seed=1, binarize=False, dequantize=False, with_label=False):
    # splits keeping tuples of (data, label) for train, val, test, 
    # aistrain and aisval partitions
    # datasets_ holding the corresponding datasets
    # data_loaders holds the loaders
    data_dir = './data'
    kwargs = {'num_workers': 12, 'pin_memory': True}
    splits = []
    data_loaders = []
    normalized = None
    shuffle = [True, False, False]
    if ais_split:
        shuffle = [True, False, False, True, False]
    
    # None dataset for synthetic target distributions
    if dataset is None or dataset == 'None':
        splits = [NoneDataset()]
        if ais_split:
            splits = splits * 5
        else:
            splits = splits * 3

    # Image datasets
    else:
        transforms_ = []

        if dataset == 'celeba':
            transforms_.append(transforms.Resize((SHAPE['celeba'][1], SHAPE['celeba'][2])))
        transforms_.append(transforms.ToTensor())

        if binarize:
            transforms_.append(transforms.Lambda(lambda x: \
                     torch.distributions.Bernoulli(probs=x).sample()))
        # dequantization not done in Huang et al https://arxiv.org/abs/2008.06653
        # but done on Wu et al. https://arxiv.org/abs/1611.04273
        elif dequantize:
            transforms_.append(transforms.Lambda(lambda x: \
                     (x + torch.rand_like(x) / 255) / (1. + 1./255)))
            mean = torch.tensor([0.4914, 0.4822, 0.4467])
            std = torch.tensor([0.2461, 0.2425, 0.2606])
        else:
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2470, 0.2435, 0.2616])
        # Normalization not done in Wu et al. https://arxiv.org/abs/1611.04273
        # but done in Huang et al https://arxiv.org/abs/2008.06653
        if dataset == 'cifar10':
            #transforms_.append(transforms.Normalize(mean, std))
            #normalized = (mean, std)
            normalized = None
        else:
            normalized = None

        transforms_ = transforms.Compose(transforms_)    

        if dataset == 'mnist':
            dset = datasets.MNIST(data_dir, train=True, download=True, 
                                  transform=transforms_)
            test_dset = datasets.MNIST(data_dir, train=False, download=True, 
                                       transform=transforms_)   

        elif dataset == 'fashionmnist':
            dset = datasets.FashionMNIST(data_dir, train=True, download=True, 
                                         transform=transforms_)
            test_dset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                              transform=transforms_)

        elif dataset == 'omniglot':
            dset = datasets.Omniglot(data_dir, background=True, download=True,
                                     transform=transforms_)
            test_dset = datasets.Omniglot(data_dir, background=False, download=True,
                                          transform=transforms_)

        elif dataset == 'cifar10':
            dset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=transforms_)
            test_dset = datasets.CIFAR10(data_dir, train=False, download=True,
                                         transform=transforms_)
        elif dataset == 'celeba':
            #splits.append(datasets.CelebA(data_dir, split="train", transform=transform, download=True))
            #splits.append(datasets.CelebA(data_dir, split="valid", transform=transform, download=True)
            #test_dset = datasets.CelebA(data_dir, split="test", transform=transform, download=True))
            splits.append(LMDBDataset(data_dir, split="train", transform=transforms_))
            splits.append(LMDBDataset(data_dir, split="val", transform=transforms_))
            test_dset = LMDBDataset(data_dir, split="test", transform=transforms_)

        else:
            raise NotImplemented

        # make validation split
        if len(splits) == 0:
            val_ratio = VALRATIO[dataset]
            np.random.seed(seed)
            indices = torch.from_numpy(np.random.choice(np.arange(len(dset)), size=(len(dset,)), replace=False))
            splits.append(Subset(dset, indices[:int((1. - val_ratio) * len(dset))]))
            splits.append(Subset(dset, indices[int((1. - val_ratio) * len(dset)):]))

        # make AIS hyperparameter training validation splits
        if ais_split:
            ais_ratio = AISRATIO[dataset]
            np.random.seed(seed)
            indices = torch.from_numpy(np.random.choice(np.arange(len(test_dset)), size=(len(test_dset,)), replace=False))
            splits.append(Subset(test_dset, indices[:int((1. - ais_ratio) * len(test_dset))]))
            splits.append(Subset(test_dset, indices[int((1. - ais_ratio) * len(test_dset)):int((1. - 0.2 * ais_ratio) * len(test_dset))]))
            splits.append(Subset(test_dset, indices[int((1. - 0.2 * ais_ratio) * len(test_dset)):]))
        else:
            splits.append(test_dset)

    # replicate last batch_size to build data_loaders
    if len(batch_sizes) < len(splits):
        batch_sizes.extend([batch_sizes[-1]] * (len(splits) - len(batch_sizes)))

    # make data_loaders with only shuffling the ones for train and aistrain
    length = []
    for i, dataset_ in enumerate(splits):     
        length.append(f'{len(dataset_)}{shuffle[i]}')
        data_loaders.append(DataLoader(dataset_, batch_size=batch_sizes[i], shuffle=shuffle[i], **kwargs))

    if dataset is not None:
        print('=> Loaded dataset {} with splits {}'.format(dataset, ','.join(length)))
    return data_loaders, normalized
    