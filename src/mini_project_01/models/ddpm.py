# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=1000):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        ### Implement Algorithm 1 here ###
        t = torch.randint(1,high=self.T,size=())
        tnet = torch.full(size=(x.shape[0],1),fill_value = t/self.T)
        eps = torch.randn(size=x.shape)
        dist = eps-self.network(torch.sqrt(self.alpha_cumprod[t])*x+torch.sqrt(1-self.alpha_cumprod[t])*eps,tnet)
        neg_elbo = torch.square(torch.linalg.vector_norm(dist,dim=1,ord=2))

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape)

        # Sample x_t given x_{t+1} until x_0 is sampled
        for t in range(self.T-1, -1, -1):
            t_inp = torch.full(size=(shape[0],),fill_value = t/self.T).reshape(-1,1)
            factor = (1-self.alpha[t])/(torch.sqrt(1-self.alpha_cumprod[t]))
            z = torch.randn(shape).to(self.alpha.device) if t > 0 else torch.full(size=shape,fill_value=0.0)
            x_t = 1/torch.sqrt(self.alpha[t])*(x_t-factor*self.network(x_t,t_inp)) + torch.sqrt(self.beta[t])*z

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()