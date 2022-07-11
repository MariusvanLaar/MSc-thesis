# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:20:13 2022

@author: Marius
"""

import datasets, models
import models
import torch
import numpy as np
import torch.nn as nn
import pickle
import time

def krons(psi):
    if psi.shape[0] == 2:
        return torch.kron(psi[0], psi[1])
    elif psi.shape[0] > 2:
        return torch.kron(psi[0], krons(psi[1:]))
    elif psi.shape[0] == 1:
        return psi[0]
    else:
        return print("Invalid input")
    
seed = 4554641
torch.manual_seed(seed)
np.random.seed(seed)
   
class ZZ_layer(nn.Module):
    "A layer applying the RZZ gate"
    def __init__(self, n_blocks: int, n_qubits: int, idx: int, weights = None):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.idx = idx
        self.weights = weights
       
    def Rz(self):
        a = -1j*(self.weights/2)
        Z = torch.Tensor([[1, -1, -1, 1]])
        return (a*Z).exp().view(-1)
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n_qubit state (2**n_qubits_perblock x 1) 
        """
        Rzs = self.Rz().cfloat()
        if self.idx == 0:
            Rzs = torch.kron(Rzs, torch.ones(2**(self.n_qubits-2)))
        elif self.idx > 0 and self.idx < self.n_qubits-1:
            I1 = torch.ones(2**self.idx)
            I2 = torch.ones(2**(self.n_qubits-2-self.idx))
            Rzs = torch.kron(I1, torch.kron(Rzs, I2))
        elif self.idx == self.n_qubits-1 or self.idx == -1:
            Rzs = torch.kron(torch.ones(2**(self.n_qubits-1)), Rzs)
        return Rzs*state

class Rx_layer(nn.Module):
    "A layer applying the Rx gate"
    def __init__(self, n_blocks: int, n_qubits: int, weights = None, weights_spread = [-np.pi/2,np.pi/2]):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread
        self.weights = weights

    def Rx(self, data=None):
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,1],[1,0]])
        if data == None:
            a = (self.weights/2).cos()
            b = (self.weights/2).sin()
        else:
            assert data.shape[1] == self.weights.nelement(), "Dimension of data incompatible"
            data = data.view((-1, *self.weights.shape[1:]))
            a = (self.weights*data/2).cos()
            b = (self.weights*data/2).sin()
        
        return a*identity - 1j*b*off_identity
        
           
    def forward(self, state, data=None):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n-qubit state (2**n x 1) 
        """
        Rxs = self.Rx(data).cfloat().view(1,2,2)
        Rxs = Rxs.repeat(self.n_qubits, 1, 1)
           
        state = torch.matmul(krons(Rxs), state) 
        return state
        

if __name__ == "__main__":
    start = time.perf_counter()
    
    n_qubits = 10
    n_blocks = 1
    idx = 0
    num_datapoints = 1000
    
    counter = 0
    X, Y = [], []
    X_h, Y_h = [], []
         
    w = torch.Tensor([-2*0.08])
    
    if idx == 0:
        Observable = torch.kron(torch.Tensor([1,-1]), torch.ones(2**(n_qubits-1)))
    elif idx == -1:
        Observable = torch.kron(torch.ones(2**(n_qubits-1)), torch.Tensor([1,-1]))
        
    Uzs = [ZZ_layer(n_blocks, n_qubits, i, weights=w) for i in range(n_qubits-1)]
    Ux = Rx_layer(n_blocks, n_qubits, weights=w)
    state = torch.zeros((2**n_qubits), dtype=torch.cfloat)
    #state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
    state[0] = 1
    for t in range(3750):
        start = time.perf_counter()
        for Uz in Uzs:
            state = Uz(state)
        state = Ux(state)
        
        
    for td in range(100):
        for Uz in Uzs:
            state = Uz(state)
        state = Ux(state)
        Y.append(torch.matmul(state.T.conj(), Observable*state))
        
    for td in range(40):
        for Uz in Uzs:
            state = Uz(state)
        state = Ux(state)
        Y_h.append(torch.matmul(state.T.conj(), Observable*state))
        
    end = time.perf_counter()
    print(end-start)
    time = np.tile(np.linspace(-1,1,num=140)*np.pi, (50,1)).T
    X = time[:100]
    X_h = time[100:]
    
    print(np.std(Y))
    
    fname=f"TIsing_{n_qubits}_{idx}" #f"datasets/data_files/TIsing_{n_qubits}_{idx}"
    synth_data = {"data": X, "labels": np.array(Y), "gen_seed": seed, "label_std": np.std(Y),
                  "X_holdout": X_h, "Y_holdout": np.array(Y_h)}
    pickling_on = open(fname+".pkl","wb")
    pickle.dump(synth_data, pickling_on)
    pickling_on.close()

import matplotlib.pyplot as plt
plt.plot(time, Y+Y_h)


