# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from layers import Rx_layer, Rz_layer, Entangle_layer
import torch 
import torch.nn as nn
import numpy as np

class TestModel(nn.Module):
    def __init__(self, n_qubits: int, batch_size: int, DC: int):
        super().__init__()   
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,2,2))
        self.Observable = self.krons(Z.repeat(n_qubits, 1, 1))
        self.Entangle = Entangle_layer([[0, 1]])
        self.Rx1 = Rx_layer([*range(n_qubits)], batch_size, weights = torch.Tensor([np.pi, 0]).reshape(1,2,1,1,1))
        self.Rx2 = Rx_layer([*range(n_qubits)], batch_size, weights = torch.Tensor([np.pi, np.pi]).reshape(1,2,1,1,1))
        
    def fRot(self, qubits, batch_size):
        return nn.Sequential(
                Rx_layer(qubits, batch_size),
                Rz_layer(qubits, batch_size),
                Rx_layer(qubits, batch_size)
                )
        
    def krons(self, psi):
        if psi.shape[0] == 2:
            return torch.kron(psi[0], psi[1])
        elif psi.shape[0] > 2:
            return torch.kron(psi[0], self.krons(psi[1:]))
        elif psi.shape[0] == 1:
            return psi[0]
        else:
            return print("Invalid input")

    def forward(self, x):
        
        state = torch.zeros((self.batch_size, self.n_qubits, 1, 2, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1 
        
        state = self.Rx1(state)
        state = self.Entangle(state)
        state = self.Rx2(state)
        state = self.Entangle(state)
        wavefunc = torch.zeros((state.shape[0], 2**state.shape[1], 1)).cdouble()
        for batch_idx in range(state.shape[0]):
            for dc_idx in range(state.shape[2]):
                wavefunc[batch_idx] = wavefunc[batch_idx].add(self.krons(state[batch_idx, :, dc_idx]))   
                
        print(wavefunc)
        
        O = torch.matmul(wavefunc.transpose(1,2).conj(), torch.matmul(self.Observable, wavefunc))
        print(O)
        return state, O

class BasicModel(nn.Module):
    def __init__(self, n_qubits: int, batch_size: int):
        super().__init__()   
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.fR0 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        self.fR1 = self.fRot([*range(n_qubits)], batch_size=batch_size) 
        self.fR2 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        # self.fR3 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        # self.fR4 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        self.fR5 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        self.fR6 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        self.fR7 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        # self.fR8 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        # self.fR9 = self.fRot([*range(n_qubits)], batch_size=batch_size)
        Z = torch.Tensor([[1,0],[0,-1]]).cdouble().reshape((1,2,2))
        self.Observable = self.krons(Z.repeat(n_qubits, 1, 1))
        #self.Entangle = Entangle_layer([[0, i] for i in range(1,DC)])
        
    def fRot(self, qubits, batch_size):
        return nn.Sequential(
                Rx_layer(qubits, batch_size),
                Rz_layer(qubits, batch_size),
                Rx_layer(qubits, batch_size)
                )
    
    def krons(self, psi):
        if psi.shape[0] == 2:
            return torch.kron(psi[0], psi[1])
        elif psi.shape[0] > 2:
            return torch.kron(psi[0], self.krons(psi[1:]))
        elif psi.shape[0] == 1:
            return psi[0]
        else:
            return print("Invalid input")
        

    def forward(self, x):
        #Can probably change this to create one large sequential model
        # then a single forward call at the end
        # ?
        state = torch.zeros((self.batch_size, self.n_qubits, 1, 2, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1 
        
        DataEncoding0 = Rx_layer([*range(self.n_qubits)], weights=x[:,0], batch_size=self.batch_size)
        DataEncoding1 = Rx_layer([*range(self.n_qubits)], weights=x[:,1], batch_size=self.batch_size)
        DataEncoding2 = Rx_layer([*range(self.n_qubits)], weights=x[:,2], batch_size=self.batch_size)
        # DataEncoding3 = Rx_layer([*range(self.n_qubits)], weights=x[:,3], batch_size=self.batch_size)
        # DataEncoding4 = Rx_layer([*range(self.n_qubits)], weights=x[:,4], batch_size=self.batch_size)
        DataEncoding5 = Rx_layer([*range(self.n_qubits)], weights=x[:,5], batch_size=self.batch_size)
        DataEncoding6 = Rx_layer([*range(self.n_qubits)], weights=x[:,6], batch_size=self.batch_size)
        DataEncoding7 = Rx_layer([*range(self.n_qubits)], weights=x[:,7], batch_size=self.batch_size)
        # DataEncoding8 = Rx_layer([*range(self.n_qubits)], weights=x[:,8], batch_size=self.batch_size)
        # DataEncoding9 = Rx_layer([*range(self.n_qubits)], weights=x[:,9], batch_size=self.batch_size)
        
        
        state = self.fR0(state)
        state = DataEncoding0(state)
        
        state = self.fR1(state)
        state = DataEncoding1(state)
        
        state = self.fR2(state)
        state = DataEncoding2(state)
        
        # state = self.fR3(state)
        # state = DataEncoding3(state)
        
        # state = self.fR4(state)
        # state = DataEncoding4(state)
        
        #state = self.Entangle(state)        
        
        state = self.fR5(state)
        state = DataEncoding5(state)
        
        state = self.fR6(state)
        state = DataEncoding6(state)
        
        state = self.fR7(state)
        state = DataEncoding7(state)
        
        # state = self.fR8(state)
        # state = DataEncoding8(state)
        
        # state = self.fR9(state)
        # state = DataEncoding9(state)    
        
        wavefunc = torch.zeros((state.shape[0], 2**state.shape[1], 1)).cdouble()
        for batch_idx in range(state.shape[0]):
            for dc_idx in range(state.shape[2]):
                wavefunc[batch_idx] = wavefunc[batch_idx].add(self.krons(state[batch_idx, :, dc_idx]))   
                
        
        O = torch.matmul(wavefunc.transpose(1,2).conj(), torch.matmul(self.Observable, wavefunc))
        return state, O
    

