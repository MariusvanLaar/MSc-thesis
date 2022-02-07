# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from layers import Rx_layer, Ry_layer, Rz_layer, Hadamard_layer, CNOT_layer, Entangle_layer, krons
import torch 
import torch.nn as nn
import numpy as np
from functools import reduce
import operator
import math

class TestModel(nn.Module):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int):
        super().__init__()   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        Z = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
        self.Observable = krons(Z.repeat(n_qubits, 1, 1))
        #If using not an all Z observable have to change forward() method below too
        self.CNOTs = CNOT_layer([*range(n_blocks)], n_qubits, [(i-1,i) for i in range(1,n_qubits)])
        #self.Rx1 = Rx_layer(batch_size, n_blocks, n_qubits)
        self.Rz2 = Rz_layer(batch_size, n_blocks, n_qubits)
        self.Entangle = Entangle_layer([((0,0), (1,0))], n_qubits)
        
    def fRot(self):
        return nn.Sequential(
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits),
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits)
                )

    def forward(self, x):
        
        state = torch.zeros((self.batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, 0, 0] = 1
        Ry = Ry_layer(self.batch_size, self.n_blocks, self.n_qubits, weights=x[0])
        Ry2 = Ry_layer(self.batch_size, self.n_blocks, self.n_qubits, weights=x[1])

        state = Ry(state)
        #state = self.Rx1(state)
        state, ent_pairs = self.Entangle(state, [])
        state = Ry2(state)
        state, ent_pairs = self.Entangle(state, [])
        #state = self.Rz2(state)
        #print(Rx)
        #state = Rx(state)
        #state, ent_pairs = self.Entangle(state, ent_pairs)
        #print(state)

        O = torch.zeros(self.batch_size,1).cdouble()
        a0 = torch.complex(torch.Tensor([1,0]), torch.Tensor([0,-1])) / ((2**0.5)*((-1j)**0.5))
        alpha = torch.Tensor([1]).cdouble()
        for j in range(int(math.log2(state.shape[2]))):
            alpha = torch.kron(alpha, a0)
        for k in range(state.shape[2]):
            a_conj = torch.roll(alpha, k).conj()
            state_conj = torch.roll(state, k, dims=2).transpose(3,4).conj()
            inn_prods = torch.matmul(state_conj, self.Observable*state)
            coefs = (a_conj*alpha).view(1,-1)
            Os = inn_prods.prod(dim=1).view(self.batch_size, -1) * coefs
            O += Os.sum(dim=1)
            print()
        
        #ent_idxs = reduce(operator.add, ent_pairs)
        #not_ent = list(set([*range(self.n_blocks)]) - set(ent_idxs))
        #O[:, not_ent] = torch.matmul(state.transpose(3,4).conj()[:, not_ent], self.Observable*state[:, not_ent])
        
        return state, O

class BasicModel(nn.Module):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights_spread: float = np.pi):
        super().__init__()   
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.batch_size = batch_size
        self.weights_spread = weights_spread
        
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(3):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.fR0 = self.fRot()
        self.fR1 = self.LAYER(1)
        copy_weights(self.fR0, self.fR1)
        self.fR2 = self.fRot()
        self.fR3 = self.LAYER(2)
        copy_weights(self.fR0, self.fR1)
        #self.fR4 = self.LAYER(1)
        self.fR5 = self.fRot()
        #self.H = Hadamard_layer(batch_size, n_blocks, n_qubits)
        Z = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
        #If using not an all Z observable have to change forward() method below too
        self.Observable = krons(Z.repeat(n_qubits, 1, 1))
        #self.Entangle = Entangle_layer([[0, i] for i in range(1,DC)])
        
        
    def LAYER(self, L):
        return nn.Sequential(
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                CNOT_layer(self.n_blocks, self.n_qubits, [(i, (i+L)%self.n_qubits) for i in range(self.n_qubits)])
                )    
    
    def fRot(self):
        return nn.Sequential(
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Rx_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Rz_layer(self.batch_size, self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                )  

    def forward(self, x):
        #Can probably change this to create one large sequential model
        # then a single forward call at the end
        # ?
                
        X = x.reshape((self.batch_size, self.n_blocks, 1, self.n_qubits, 1))
        
        Ry_data = Ry_layer(self.batch_size, self.n_blocks, self.n_qubits, weights=X)

        
        state = torch.zeros((self.batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)

        state = self.fR0(state)
        
        state = Ry_data(state)
        
        state = self.fR1(state)
        
        state = Ry_data(state)
        
        state = self.fR2(state)
        
        state = Ry_data(state)

        state = self.fR3(state)
        
        #state = Ry_data(state)
        
        #state = self.fR4(state)
                
        state = self.fR5(state)
                
        O = torch.matmul(state.transpose(3,4).conj(), self.Observable*state)
        return state, O
    

