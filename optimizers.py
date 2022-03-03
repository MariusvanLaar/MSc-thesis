# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:22:37 2022

@author: Marius
"""

import torch
import torch.optim as opt
from torch.optim import Optimizer


class SPSA(Optimizer):
    """Implements the SPSA optimizer"""
    
    def __init__(self, params, lr=1):
        
        defaults = dict(lr=lr, a=0.01, alpha=0.01, gamma=0.02, epoch=0)
        #num_params = len(params)
        
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
        loss = None
        for group in self.param_groups:

            lr = group['lr']
            epoch = group['epoch']
            gamma = group['gamma']
            a = group['a']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is not None:
                    
                    pertb_vector = 2*torch.randint_like(p, low=0, high=2) - 1
                    p.add_(lr*pertb_vector)
                    with torch.enable_grad():
                        l1 = closure()
                    p.add_(-2*lr*pertb_vector)
                    with torch.enable_grad():
                        l2 = closure()
                    #g = (l1-l2) / 2*lr*pertb_vector
                    if l1 <= l2:
                        g = -pertb_vector
                    else:
                        g = pertb_vector
                    p.add_(lr*pertb_vector - a*g)
                    
            #group['a'] = a/(1+epoch)**alpha
            #group['lr'] = lr/(1+epoch)**gamma
            #group['epoch'] += 1

            state = self.state[p]
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        