# -*- coding: utf-8 -*-
"""
Created on Sat May 14 15:50:06 2022

@author: Marius
"""

"""Idea is to create different classes of observables and add this as a user input
    in the main program. The inputs for each class are n_qubits, n_blocks, 
    block and qubit index if going for single qubit index.
    User input will be observable class and return_probability (bool) to
    indictate whether to map the input to the range [0,1]"""

import torch
import torch.nn as nn
import numpy as np
import math

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
        return O.real.float()
    
    def return_probability(self, state):
        return torch.clamp(0.5*(self.exp_val(state)+1), min=0, max=1)
    #Clamp is necessary for over or underflow errors leading to values just above 1 or below 0 (on the scale 0f e-08)
    #Can be a source of bugs though, especially when introducing new features "upstream" the clamp should be disabled.
