# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:22:37 2022

@author: Marius
"""

import torch
import numpy as np
from torch.optim import Optimizer
from numpy import pi

import torch.nn as nn



class SPSA(Optimizer):
    """Implements the SPSA optimizer"""
    
    def __init__(self, params, lr=1):
        
        defaults = dict(lr=lr, a=10, A=10,  alpha=0.612, gamma=1/6, epoch=0)
        
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super(SPSA, self).__setstate(state)
        
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        def clip(x, bounds=[-0.2, 0.2]):
            return torch.clamp(x, min=bounds[0], max=bounds[1])
        
        loss = None
        for group in self.param_groups:

            lr = group['lr'] / (group['epoch'] + 1)**group['gamma']
            a = group['a'] / (group['A'] + group["epoch"] + 1)**group['alpha']

            p = nn.utils.parameters_to_vector(group["params"])
            pertb_vector = torch.randint_like(p, high=1)*2 - 1
            
            p.add_(lr*pertb_vector)
            nn.utils.vector_to_parameters(p, group["params"])
            l1 = closure()
            p.add_(-2*lr*pertb_vector)
            nn.utils.vector_to_parameters(p, group["params"])
            l2 = closure()
            g = (l1-l2) / 2*lr*pertb_vector
            
            p.add_(lr*pertb_vector - clip(a*g))
            nn.utils.vector_to_parameters(p, group["params"])
            
            group['epoch'] += 1

            state = self.state[p]
        if closure is not None:
            loss = closure()
        
class CMA(Optimizer):
    """Implements the CMA optimizer"""
    
    def __init__(self, params, lr=1, popsize=40, s0=pi/6, mu=0.2):
        import cma
        defaults = dict(bounds=[-pi, pi], popsize=popsize, CMA_cmean=lr, CMA_mu=int(popsize*mu))
        
        super().__init__(params, defaults)
        
        x = torch.cat([x.view(-1) for x in list(self.param_groups)[0]["params"]]).detach()
        x0 = torch.zeros_like(x)
        sigma0 = s0
        
        self.cma_opt = cma.CMAEvolutionStrategy(x0, sigma0, inopts=defaults)
        
    def __setstate__(self, state):
        super(CMA, self).__setstate(state)
        
    def set_params_(self, candidate_p):
        for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    size_p = p.nelement()
                    shape_p = p.shape
                    p.add_(-p)
                    p.add_(torch.tensor(candidate_p[i*size_p:(i+1)*size_p]).view(shape_p))
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        losses = []
        xs = self.cma_opt.ask()

        for x_cand in xs:
            self.set_params_(x_cand)
            if closure is not None:
                loss = closure()
                losses.append(loss.item())

        #Tell optimizer results of each candidate solution
        self.cma_opt.tell(xs, losses) 
        #Set model parameters to best candidate
        self.set_params_(self.cma_opt.best.x)
        #Return mean loss of all candidates
        loss = np.mean(losses)
        
class CWD(Optimizer):
    """Implements coordinate wise descent"""
    
    def __init__(self, params, **kwargs):
        
        defaults = {}
        self.steps = 30
        super().__init__(params, defaults)
        self.dru_prange = torch.linspace(0, 2, steps=self.steps)
        self.var_prange = torch.linspace(-np.pi, np.pi, steps=self.steps)
        
        for group in self.param_groups:
            P = nn.utils.parameters_to_vector(group["params"])
            self.mask = (P==1)
    
    def __setstate__(self, state):
        super(CWD, self).__setstate(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        
        for group in self.param_groups:
            P = nn.utils.parameters_to_vector(group["params"])
            for j in range(len(P)):
                if not self.mask[j]:
                    p_range = self.var_prange 
                else:
                    p_range = self.dru_prange
                    
                losses = []
                    
                for p_val in p_range:
                    P[j] = p_val
                    nn.utils.vector_to_parameters(P, group["params"])
                    
                    losses.append(closure())
                    
                best_idx = torch.argmin(torch.Tensor(losses))
                P[j] = p_range[best_idx]

        loss = torch.min(torch.Tensor(losses))

        
        
        
        
                