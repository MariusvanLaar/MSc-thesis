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
        self.fR1 = self.fRot([*range(n_qubits)], batch_size=batch_size) 
        self.fR2 = self.fRot([*range(n_qubits)], batch_size=batch_size) 
        Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,1,1,2,2))
        self.Observable = Z.repeat((batch_size, n_qubits, 1, 1, 1))
        self.Entangle_1 = Entangle_layer([(0,1), (1,2)])
        self.Entangle_2 = Entangle_layer([(1,2)])
        
    def fRot(self, qubits, batch_size):
        return nn.Sequential(
                Rx_layer(qubits, batch_size),
                Rz_layer(qubits, batch_size),
                Rx_layer(qubits, batch_size)
                )
        

    def forward(self, x):
        #Can probably change this to create one large sequential model
        # then a single forward call at the end
        # ?
        state = torch.zeros((self.batch_size, self.n_qubits, 1, 2, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1 
        #Rx1 = Rx_layer([*range(self.n_qubits)], weights=x, batch_size=self.batch_size)
        #state = self.fR1(state)
        #state = Rx1(state)
        
        state = self.Entangle_1(state)
        #state = self.Entangle_2(state)
        
        #state = self.fR2(state)

        #state = self.fRx2(state)
        #state = Rx1(state)
        #state = self.fRx3(state)
        #state = Rx1(state)
        
        #state = Entangle_layer([(3,4), (5,6), (8,6)])(state)
        
        #state = self.fRx4(state)
        #state = Rx1(state)
        #state = self.fRx5(state)
        #state = Rx1(state)
        #state = self.fRz[j](state)
        
        O = torch.matmul(state.transpose(3,4).conj(), torch.matmul(self.Observable, state))
        return state, O.mean(dim=2)
    

