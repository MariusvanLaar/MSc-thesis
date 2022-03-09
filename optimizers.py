# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:22:37 2022

@author: Marius
"""

import torch
import torch.optim as opt
import numpy as np
from torch.optim import Optimizer
from numpy import pi
import cma


class SPSA(Optimizer):
    """Implements the SPSA optimizer"""
    
    def __init__(self, params, lr=1):
        
        defaults = dict(lr=lr, a=1e4, alpha=0.99, gamma=0.99, epoch=0)
        
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
        def clip(x, bounds=[-0.1, 0.1]):
            return torch.clamp(x, min=bounds[0], max=bounds[1])
        
        loss = None
        for group in self.param_groups:

            lr = group['lr']
            gamma = group['gamma']
            a = group['a']
            alpha = group['alpha']

            for p in group['params']:
                
                pertb_vector = 2*torch.randint_like(p, low=0, high=2) - 1
                p.add_(lr*pertb_vector)
                l1 = closure()
                p.add_(-2*lr*pertb_vector)
                l2 = closure()
                g = (l1-l2) / 2*lr*pertb_vector
                # if l1 <= l2:
                #     g = -lr*pertb_vector
                # else:
                #     g = lr*pertb_vector
                
                p.add_(lr*pertb_vector - clip(a*g))
                    
            group['a'] = a*alpha
            group['lr'] = lr*gamma
            group['epoch'] += 1

            state = self.state[p]
        if closure is not None:
            loss = closure()
        
class CMA(Optimizer):
    """Implements the CMA optimizer"""
    
    def __init__(self, params, lr=1):
    
        defaults = dict(bounds=[-pi, pi], popsize=40, CMA_cmean=lr)
        
        super().__init__(params, defaults)
        
        x = torch.cat([x.view(-1) for x in list(self.param_groups)[0]["params"]]).detach()
        x0 = torch.zeros_like(x)
        sigma0 = pi/2
        
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
                