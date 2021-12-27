# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from layers import Rx_layer, Rz_layer
import torch 
import torch.nn as nn
import numpy as np

class BasicModel(nn.Module):
    def __init__(self, n_qubits):
        super().__init__()   
        self.n_qubits = n_qubits
        self.fRx = Rx_layer([*range(n_qubits)])
        Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,2,2))
        self.Observable = Z.repeat((n_qubits, 1, 1))
        #self.Rx1 = Rx_layer([0,1], torch.Tensor())
        #self.Rx2 = Rx_layer([0,1], torch.Tensor(self.weights))

    def forward(self, x):
        state = torch.zeros((self.n_qubits, 2, 1), dtype=torch.cdouble)
        state[:, 0, 0] = 1 
        Rx1 = Rx_layer([0,1], weights=x)
        state = Rx1(state)
        state = self.fRx(state)
        O = torch.bmm(state.transpose(1,2).conj(), torch.bmm(self.Observable, state))
        return state, O
    

