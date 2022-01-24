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
I = torch.eye(2)

def CNOT(pair, n_qubits):
    if pair[0] > pair[1]:
        U = torch.zeros((2**n_qubits, 2**n_qubits))
        I = torch.eye(2**(n_qubits-1)).flatten()
        stride = 2**(n_qubits-pair[0]-1)
        unstride = 2**(pair[0])
        mask = torch.Tensor([[1,0],[0,0]]).bool()
        mask = mask.repeat_interleave(stride, dim=0).repeat_interleave(stride, dim=1)
        mask = mask.repeat(unstride, unstride)
        U[mask] = I
        
        I = torch.eye(2**(n_qubits-2)).flatten()
        outer_mask_1 = torch.Tensor([[0,0], [1,0]]).bool()
        outer_mask_2 = torch.Tensor([[0,1], [0,0]]).bool()
        inner_mask = torch.Tensor([[0,0],[0,1]]).bool()
        inner_mask = inner_mask.repeat_interleave(stride, dim=0).repeat_interleave(stride, dim=1)
        inner_mask = inner_mask.repeat(unstride//2, unstride//2)
        mask = torch.kron(outer_mask_1, inner_mask)
        U[mask] = I
        mask = torch.kron(outer_mask_2, inner_mask)
        U[mask] = I

    elif pair[0] < pair[1]:
        X = torch.Tensor([[0,1],[1,0]])
        I1 = torch.eye(2**(pair[1] - pair[0] - 1))
        I2 = torch.eye(2**(n_qubits - pair[1] - 1))
        M = torch.kron(I1, torch.kron(X, I2))
        U = torch.eye(M.shape[0]*2)
        U[U.shape[0]//2:, U.shape[0]//2:] = M
        pre_I = torch.eye(2**pair[0])
        U = torch.kron(pre_I, U)
        
    else:
        raise RuntimeError("Invalid qubit pair given")
        
    return U
        
        

def apply(U, state):
    print(torch.matmul(U, state))
    
qubits = 4
state = torch.zeros((2**qubits, 1))
state[8] = 1
print(state)

U1 = CNOT((0,1), qubits)
U2 = CNOT((1,2), qubits)
print(torch.matmul(U1, U1))
print()
# apply(U1,state)
# apply(torch.matmul(U2, U1),state)






