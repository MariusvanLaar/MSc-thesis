# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from layers import Rx_layer, Rz_layer, Hadamard_layer, CNOT_layer, Entangle_layer
import torch 
import torch.nn as nn
import numpy as np

class TestModel(nn.Module):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int):
        super().__init__()   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        Z = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
        self.Observable = self.krons(Z.repeat(n_qubits, 1, 1))
        #If using not an all Z observable have to change forward() method below too
        self.CNOTs = CNOT_layer([*range(n_blocks)], n_qubits, [(i-1,i) for i in range(1,n_qubits)])
        #self.Rx1 = Rx_layer(batch_size, n_blocks, n_qubits)
        self.Rz2 = Rz_layer(batch_size, n_blocks, n_qubits)
        
    def fRot(self):
        return nn.Sequential(
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits)
                )
    
    def krons(self, psi):
        if psi.shape[0] > 2:
            return torch.kron(psi[0], self.krons(psi[1:]))
        elif psi.shape[0] == 2:
            return torch.kron(psi[0], psi[1])
        elif psi.shape[0] == 1:
            return psi[0]
        else:
            return print("Invalid input")

    def forward(self, x):
        
        state = torch.zeros((self.batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1
        
        #state = self.Rx1(state)
        #state = self.Entangle(state)
        #state = self.Rz2(state)
        Rx = Rx_layer(self.batch_size, self.n_blocks, self.n_qubits, weights=x)
        #print(Rx)
        state = Rx(state)
        print(state)
        state = self.CNOTs(state)
        #print(state)
        O = torch.matmul(state.transpose(3,4).conj(), self.Observable*state)
        return state, O

class BasicModel(nn.Module):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int):
        super().__init__()   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.fR0 = self.fRot()
        self.fR1 = self.fRot() 
        self.fR2 = self.fRot()
        # self.fR3 = self.fRot()
        # self.fR4 = self.fRot()
        self.fR5 = self.fRot()
        self.fR6 = self.fRot()
        self.fR7 = self.fRot()
        self.H = Hadamard_layer(batch_size, n_blocks, n_qubits)
        Z = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
        #If using not an all Z observable have to change forward() method below too
        self.Observable = self.krons(Z.repeat(n_qubits, 1, 1))
        #self.Entangle = Entangle_layer([[0, i] for i in range(1,DC)])
        
    def fRot(self):
        return nn.Sequential(
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits)
                )

    def krons(self, psi):
        if psi.shape[0] > 2:
            return torch.kron(psi[0], self.krons(psi[1:]))
        elif psi.shape[0] == 2:
            return torch.kron(psi[0], psi[1])
        elif psi.shape[0] == 1:
            return psi[0]
        else:
            return print("Invalid input")
    

    def forward(self, x):
        #Can probably change this to create one large sequential model
        # then a single forward call at the end
        # ?
        state = torch.zeros((self.batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, -1, 0] = 1 
        
        state = self.H(state)
        # state = self.fR0(state)
        
        # state = self.fR1(state)
        
        # state = self.fR2(state)
        
        # state = self.fR3(state)
        
        # state = self.fR4(state)
        
        #state = self.Entangle(state)        
        
        # state = self.fR5(state)
        
        # state = self.fR6(state)
        
        # state = self.fR7(state)
        
        # state = self.fR8(state)
        
        # state = self.fR9(state)
     
        
        O = torch.matmul(state.transpose(3,4).conj(), self.Observable*state)
        return state, O
    

