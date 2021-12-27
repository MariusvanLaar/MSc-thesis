# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy.linalg import schur, expm
import torch

def Rz(self):
    L = len(self.weights)
    a = 1j*(self.weights/2).view(-1,1,1)
    Z = torch.Tensor([[1,-1]]).reshape(1,2,2,).repeat(L,1,1,)
    print(a*Z)
    return (a*Z).exp()
    

weights = torch.Tensor([0,np.pi,2*np.pi])
L = len(weights)
Z = torch.Tensor([[-1,1]]).reshape(1,2,).repeat(L,1,)
print((weights.view(3,1)*Z).exp())
print(torch.diag_embed((weights.view(3,1)*Z).exp(), dim1=1, dim2=2))

    

    




