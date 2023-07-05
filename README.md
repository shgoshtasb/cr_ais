# Adaptive Annealed Importance Sampling with Contant Rate Progress

The repository contains the code of the paper 'Adaptive Annealed Importance Sampling with Constant Rate Progress' paper.

In this work, we show the connection between commonly used annealing path heuristics to the divergence of particle distribution and the target in Annealed Importance Sampling algorithm. More specifically we show tht the geometric mean path is achieved by modifying the initial density along the functional derivative of the inverse KL-divergence with infinitesimal step size. We further show a similar connection between the recently introduced [power mean paths](https://arxiv.org/abs/2012.07823) and $\alpha$-divergences and derive an Ordinary Differential Equation describing the annealing dynamics corresponding to $f$-divergences.

Relying on this connection we can adapt the annealing schedule to reduce the divergence equally in each step and establish an annealing sequence where each step has a similar contribution to the convergence of particle distribution to the target. The tuning procedure avoids expensive searching loops and unreliable ESS/CESS based schedule updates in traditional adaptive AIS algorithm and can be implemented with a simple empirical variance estimation. Constant Rate schedule provides better sample dispersion in the experiments and more efficient log normalization factor estimation than heuristic scheduled or adaptive AIS, individually.

# How to run?

The experiments which appear in the paper can be run via the `run.py` script. For example:

```bash
python run.py --sampler CR --transition HMC --transition_step_size 0.5 --hmc_alpha 1. --hmc_partial_refresh 10 --hmc_n_leapfrogs 1 --transition_update fixed --n_samples 2048 --test_n_samples 4096 --testname 1ec34c --target UdNormal_128 --path power --annealing_alpha 0.0 --max_M 2048 --tol 1e-3 --min_step=0.0000001 --seed 1 --latent_dim 128  --dt 1.0
```

- **sampler** argument can be replaced with 'Plain', 'Adaptive', 'SMC', 'AdaptiveSMC', 'CRSMC', 'MCD'

- **target** argument can be replaced with 'UdNormal_[128|512]', 'UdMixture_[128|512]', 'UdLaplace_[128|512]', 'UdStudentT_[128|512]', 'pima', 'sonar'

    For relevant arguments see `utils/experiments.py` 

For the VAE experiments first train a VAE on MNIST dataset using 

```bash
python train_model.py  --testname 1ec34c/models --model vae --latent_dim 50 --net binary --dataset mnist --binarize
```
then run

```bash
python run_vae.py --sampler CR --dataset mnist --transition HMC --transition_step_size 0.5 --hmc_alpha 1. --hmc_partial_refresh 10 --hmc_n_leapfrogs 1 --transition_update fixed --n_samples 16 --test_n_samples 16 --testname 1ec34c --target vae50  --path power --annealing_alpha 0.0 --max_M 2048 --tol 1e-3 --min_step=0.0000001 --seed 1 --latent_dim 50  --dt 512.0
```

# Citation

The original paper can be found [here](). 



