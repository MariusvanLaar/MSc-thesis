# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 17:22:37 2022

@author: Marius
"""

import torch
import torch.optim as opt
import numpy as np
from torch.optim import Optimizer


class SPSA(Optimizer):
    """Implements the SPSA optimizer"""
    
    def __init__(self, params, lr=1):
        
        defaults = dict(lr=lr, a=1e5, alpha=0.999, gamma=0.999, epoch=0)
        
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
        