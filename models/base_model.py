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
import ast


class BaseModel(nn.Module):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2],
                 observable="Final"):
        super().__init__()   
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread
        self.block_obs = None
        self.observables = {"All": self.all_qubit_Z, "Final": self.single_qubit_Z}
        assert type(observable) == str, "Invalid observable given"
        if observable in self.observables:
            self.observables[observable]()
        else: #list of [block idx, qubit idx] 
            assert len(ast.literal_eval(observable)) <= 2, "Invalid single qubit observable observable given"
            self.single_qubit_Z(ast.literal_eval(observable))
        
        
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

    def ZYfRot(self, weights_spread=False):
        if not weights_spread:
            weights_spread = self.weights_spread
        return nn.Sequential(
                Rz_layer(self.n_blocks, self.n_qubits, weights_spread=weights_spread),
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
        
    def single_qubit_Z(self, idxs=[-1]):
        """idxs is a list of [block index, qubit index] to be the target single qubit observable.
            If the block index is omitted, the observable is Z on the qubit index in each
            block (which is strictly not a single qubit observable but we make do)."""
        assert idxs[-1] < self.n_qubits, "Invalid qubit idx given"
        #This section initializes a Pauli Z observable on the idx qubit of each block
        idx = idxs[-1]
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
        #This section customizes the above observable to act on a specific block
        if len(idxs) == 2:
            assert idxs[0] < self.n_blocks, "Invalid qubit idx given"
            self.Observable = self.Observable.repeat(self.n_blocks, 1, 1, 1)
            nidx = [i for i in range(self.n_blocks) if i !=idxs[0]]
            I = torch.ones((self.n_blocks-1, 1, 2**self.n_qubits, 1))
            self.Observable[nidx] = I
                
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
    
    
    def return_probability(self, output):
        return torch.clamp(0.5*(output+1), min=0, max=1)
        #Clamp is necessary for over or underflow errors leading to values just above 1 or below 0 (on the scale 0f e-08)
        #Can be a source of bugs though, especially when introducing new features "upstream" the clamp should be disabled.
    
    def exp_val_(self, state):
        """
        Alternative computation for calculating the expectation value.
        This function is MUCH slower than the above but uses less inplace memory.
        It uses a loop over the batch size and over the divide and conquer dimension
        in an attempt to be more memory efficient.
        Unfortunately for gradient based methods the computational graph is still too large
        and causes a crash for the same system sizes as the above function. 
        Only for gradient free optimization and very large systems will this function when the above won't.
        At this point however the computation will be unreasonably slow anyway so consider this function 
        more of a christmas decoration.
        """
        batch_size = state.shape[0]
        O = torch.zeros(batch_size, self.n_blocks, state.shape[2], state.shape[2]).cfloat()
        a0 = torch.complex(torch.Tensor([1,0]), torch.Tensor([0,-1])) / ((2**0.5)*(-1j)**0.5)
        alpha = torch.Tensor([1]).cfloat()
        for j in range(int(math.log2(state.shape[2]))):
            alpha = torch.kron(alpha, a0)
        alpha = alpha.conj().view(-1,1)*alpha.view(1,-1)
        for b in range(batch_size): #Loop over blocks
            for d1 in range(state.shape[2]): #Loop over d&c dim
                for d2 in range(d1, state.shape[2]): #Loop over d&c dim
                    state_conj = state[b, :, d1].transpose(1,2).conj()
                    inn_prod = torch.matmul(state_conj, self.Observable*state[b,:,d2]).view(-1)
                    O[b, :, d1, d2] = inn_prod
        O = O + torch.triu(O, diagonal=1).conj().transpose(2,3)
        O = O.prod(dim=1)
        O = O*alpha
        O = O.sum(dim=[1,2])
           
        return O.real.float()