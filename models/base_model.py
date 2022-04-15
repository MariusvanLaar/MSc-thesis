# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:35:20 2022

@author: Marius
"""

import torch
import torch.nn as nn
import numpy as np
import math
from .layers import Rx_layer, Ry_layer, Rz_layer, Hadamard_layer, CNOT_layer, Entangle_layer, krons



class BaseModel(nn.Module):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = [-np.pi/2,np.pi/2], grant_init: bool = False):
        super().__init__()   
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread
        
        Observ = torch.Tensor([[1,-1]]).cfloat().reshape((1,2,1))
        Observ = Observ.repeat(n_qubits,1,1)
        #Observ[-1, 1] = -1
        #If using not a diagonal observable have to change exp_val() method below too
        self.Observable = krons(Observ)
        
    def cnot_(self, offset, leap):
        return nn.Sequential(
                CNOT_layer(self.n_blocks, self.n_qubits, 
                           [((i+offset)%self.n_qubits, (i+offset+leap)%self.n_qubits) for i in range(self.n_qubits)])
                )    
    
    def AfRot(self, weights_spread=False):
        if not weights_spread:
            weights_spread = self.weights_spread
        return nn.Sequential(
                Rz_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
                Ry_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
                Rz_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
                )  
    
    def YfRot(self, weights_spread=False):
        if not weights_spread:
            weights_spread = self.weights_spread
        return nn.Sequential(
                Ry_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
                )

    def XfRot(self, weights_spread=False):
        if not weights_spread:
            weights_spread = self.weights_spread
        return nn.Sequential(
                Rx_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
                )
        
    def copy_weights(self, control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
                
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")
        
    def single_qubit_Z(self, idx):
        """idx is the index of the qubit within the block to be measured"""
        if idx == 0:
            self.Observable = torch.kron(torch.Tensor([1,-1]), torch.ones(2**(self.n_qubits-1))).view(-1,1)
        elif idx > 0 and idx < self.n_qubits-1:
            I1 = torch.ones(2**idx)
            I2 = torch.ones(2**(self.n_qubits-1-idx))
            self.Observable = torch.kron(I1,
                                         torch.kron(torch.Tensor([1,-1]),
                                                    I2)).view(-1,1)
        elif idx == self.n_qubits-1 or idx == -1:
            self.Observable = torch.kron(torch.ones(2**(self.n_qubits-1)), torch.Tensor([1,-1])).view(-1,1)
            
    def all_qubit_Z(self):
        """Quick function to revert an observable to all-Pauli-Z"""
        Observ = torch.Tensor([[1,-1]]).cfloat().reshape((1,2,1))
        Observ = Observ.repeat(self.n_qubits,1,1)
        self.Observable = krons(Observ)

    def exp_val_(self, state):
        batch_size = state.shape[0]
        O = torch.zeros(batch_size, self.n_blocks, state.shape[2], state.shape[2]).cfloat()
        a0 = torch.complex(torch.Tensor([1,0]), torch.Tensor([0,-1])) / ((2**0.5)*(-1j)**0.5)
        alpha = torch.Tensor([1]).cfloat()
        for j in range(int(math.log2(state.shape[2]))):
            alpha = torch.kron(alpha, a0)
        #alpha = alpha.conj().view(-1,1)*alpha.view(1,-1)
        for b in range(self.n_blocks): #Loop over blocks
            for d1 in range(state.shape[2]): #Loop over d&c dim
                for d2 in range(d1, state.shape[2]): #Loop over d&c dim
                    state_conj = state[:, b, d1].transpose(1,2).conj()
                    inn_prod = torch.matmul(state_conj, self.Observable*state[:,b,d2]).view(-1)
                    O[:, b, d1, d2] = inn_prod
        O = O + torch.triu(O, diagonal=1).conj().transpose(2,3)
        O = O.prod(dim=1)
        O = O*alpha
        O = O.sum(dim=[1,2])
           
        return torch.clamp(0.5*(O.real.float()+1), min=0, max=1)
        
    def exp_val(self, state):
        batch_size = state.shape[0]
        O = torch.zeros(batch_size).cfloat()
        a0 = torch.complex(torch.Tensor([1,0]), torch.Tensor([0,-1])) / ((2**0.5)*(-1j)**0.5)
        alpha = torch.Tensor([1]).cfloat()
        for j in range(int(math.log2(state.shape[2]))):
            alpha = torch.kron(alpha, a0)
        for k in range(state.shape[2]):
            a_conj = torch.roll(alpha, k).conj()
            state_conj = torch.roll(state, k, dims=2).transpose(3,4).conj()
            inn_prods = torch.matmul(state_conj, self.Observable*state)
            coefs = (a_conj*alpha).view(1,-1)
            Os = inn_prods.prod(dim=1).view(batch_size, -1) * coefs
            
            O += Os.sum(dim=1)
           
        return torch.clamp(0.5*(O.real.float()+1), min=0, max=1)
    #Clamp is necessary for over or underflow errors leading to values just above 1 or below 0 (on the scale 0f e-08)
    #Can be a source of bugs though, especially when introducing new gate types.