# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi

Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
I = np.array([[1,0],[0,1]])

def Rx(theta):
    a = (theta/2).cos()
    b = (theta/2).sin()
    identity = torch.eye(2)
    off_identity = torch.Tensor([[0,1],[1,0]])
    return a*identity - 1j*b*off_identity

U = torch.Tensor(np.kron(X,X)).cdouble()
U += 1j*torch.Tensor(np.kron(Z,Z))
print(U)

for i in range(3):
    Ur = torch.kron(Rx(torch.rand(1)), Rx(torch.rand(1))).cdouble()
    t1 = torch.mm(U, Ur)
    t2 = torch.mm(Ur, U)
    print(t1-t2)


def next_batch():
        
    def threshold(x):
        if x > pi:
            return -1
        else:
            return 1
        
    X = torch.rand((1,2,1,1,1))*2*pi
    angles = X.sum(dim=1) % 2*pi
    Y = angles.apply_(threshold).cdouble()
    #Y = torch.rand((self.batch_size, self.n_qubits,1,1)).cdouble()*2 -1
    return X, Y

# Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,1,2,2))
# Observable = Z.repeat((1, 2, 1, 1))
# x, y = next_batch()
# print(x, y)
# state = torch.zeros((1, 2, 2, 1), dtype=torch.cdouble)
# state[:, :, 0, 0] = 1
# #print(state.shape, Rx(x.sum(dim=1)).shape)

# state = torch.matmul(Rx(x.sum(dim=1)).cdouble(), state)
# O = torch.matmul(state.transpose(2,3).conj(), torch.matmul(Observable, state))
# print(O)

