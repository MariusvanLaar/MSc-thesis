# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from layers import Rx_layer, Rz_layer, Entangle_layer
import torch 
import torch.nn as nn
import numpy as np

class BasicModel(nn.Module):
    def __init__(self, n_qubits: int, batch_size: int):
        super().__init__()   
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.fRx = [Rx_layer([*range(n_qubits)], batch_size=batch_size) for i in range(5)]
        self.fRz = [Rz_layer([*range(n_qubits)], batch_size=batch_size) for i in range(5)]
        Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,1,1,2,2))
        self.Observable = Z.repeat((batch_size, n_qubits, 1, 1, 1))
        self.Entangle = Entangle_layer([(0,1)])
        

    def forward(self, x):
        state = torch.zeros((self.batch_size, self.n_qubits, 1, 2, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1 
        Rx1 = Rx_layer([*range(self.n_qubits)], weights=x, batch_size=self.batch_size)
        for j in range(5):
            state = self.fRx[j](state)
            state = Rx1(state)
            state = self.fRz[j](state)
        #state = self.Entangle(state)
        O = torch.matmul(state.transpose(3,4).conj(), torch.matmul(self.Observable, state))
        return state, O.mean(dim=2)
    

