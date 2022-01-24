# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:32:13 2021

@author: Marius
"""

import torch
import torch.nn as nn
import numpy as np

def krons(psi):
    if psi.shape[0] == 2:
        return torch.kron(psi[0], psi[1])
    elif psi.shape[0] > 2:
        return torch.kron(psi[0], krons(psi[1:]))
    elif psi.shape[0] == 1:
        return psi[0]
    else:
        return print("Invalid input")

class Rx_layer(nn.Module):
    "A layer applying the Rx gate"
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights = None):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.batch_size = batch_size   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(self.batch_size, self.n_blocks,1,self.n_qubits,1,1))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
            self.weights.cdouble()
        else:
            if weights.shape[0] == batch_size and weights.shape[1] == n_blocks and weights.shape[3] == n_qubits:
                self.weights = weights     
            elif weights.shape[3] == 1 and self.n_qubits > 1: 
                self.weights = weights.repeat(1,1,1,self.n_qubits,1)
            else:
                raise RuntimeError("Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count")
        

    def Rx(self):
        a = (self.weights/2).cos()
        b = (self.weights/2).sin()
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,1],[1,0]])
        return a*identity - 1j*b*off_identity
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n-qubit state (2**n x 1) 
        """
        
        Rxs = self.Rx().cdouble()
        U = torch.zeros(*Rxs.shape[:3], 2**self.n_qubits, 2**self.n_qubits).cdouble()
        for batch_idx in range(self.batch_size):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rxs[batch_idx, block_idx, 0])
            
        state = torch.matmul(U, state) 

        return state
        
    ### Consider how to include nograd if weights is given. This might be already incorporated?

class Rz_layer(nn.Module):
    "A layer applying the Rz gate"
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights = None):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.batch_size = batch_size   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(self.batch_size, self.n_blocks,1,self.n_qubits,1))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
            self.weights.cdouble()
        else:
            if weights.shape[0] == batch_size and weights.shape[1] == n_blocks and weights.shape[3] == n_qubits:
                self.weights = weights     
            elif weights.shape[3] == 1 and self.n_qubits > 1:
                self.weights = weights.repeat(1,1,1,self.n_qubits,1)
            else:
                raise RuntimeError("Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count")

    def Rz(self):
        a = -1j*(self.weights/2)
        Z = torch.Tensor([[1, -1]])
        return (a*Z).exp() 
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n_qubit state (2**n_qubits_perblock x 1) 
        """
        Rzs = self.Rz().cdouble()
        U = torch.zeros(*Rzs.shape[:3], 2**self.n_qubits, 1).cdouble()
        for batch_idx in range(self.batch_size):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rzs[batch_idx, block_idx, 0]).reshape(-1,1) #Reshape because using flattend matrix
            
        state = U*state 

        return state
    
class Hadamard_layer(nn.Module):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int):        
        super().__init__()
        
        self.H = torch.Tensor([[1, 1], [1, -1]]).reshape(1,2,2)/(2**0.5)
        self.H = krons(self.H.repeat(n_qubits,1,1)).cdouble()
        
    def forward(self, state):
        return torch.matmul(self.H, state)
    
    
class CNOT_layer(nn.Module):
    def __init__(self, block_ids: list, block_size: int, qubit_pairs: list):        
        super().__init__()
        
        self.block_ids = block_ids
        self.block_size = block_size
        self.pairs = qubit_pairs
        
    def gen_CNOT(self, pair): 
        if pair[0] > pair[1]:
            U = torch.zeros((2**self.block_size, 2**self.block_size))
            I = torch.eye(2**(self.block_size-1)).flatten()
            stride = 2**(self.block_size-pair[0]-1)
            unstride = 2**(pair[0])
            mask = torch.Tensor([[1,0],[0,0]]).bool()
            mask = mask.repeat_interleave(stride, dim=0).repeat_interleave(stride, dim=1)
            mask = mask.repeat(unstride, unstride)
            U[mask] = I
            
            I = torch.eye(2**(self.block_size-2)).flatten()
            outer_mask_1 = torch.Tensor([[0,0], [1,0]]).bool()
            outer_mask_2 = torch.Tensor([[0,1], [0,0]]).bool()
            inner_mask = torch.Tensor([[0,0],[0,1]]).bool()
            inner_mask = inner_mask.repeat_interleave(stride, dim=0).repeat_interleave(stride, dim=1)
            inner_mask = inner_mask.repeat(unstride//2, unstride//2)
            mask = torch.kron(outer_mask_1, inner_mask)
            U[mask] = I
            mask = torch.kron(outer_mask_2, inner_mask)
            U[mask] = I
    
        elif pair[0] < pair[1]:
            X = torch.Tensor([[0,1],[1,0]])
            I1 = torch.eye(2**(pair[1] - pair[0] - 1))
            I2 = torch.eye(2**(self.block_size - pair[1] - 1))
            M = torch.kron(I1, torch.kron(X, I2))
            U = torch.eye(M.shape[0]*2)
            U[U.shape[0]//2:, U.shape[0]//2:] = M
            pre_I = torch.eye(2**pair[0])
            U = torch.kron(pre_I, U)
            
        else:
            raise RuntimeError("Invalid qubit pair given")
            
        return U
    
    def forward(self, state):
        for pair in self.pairs:
            CNOT = self.gen_CNOT(pair).cdouble()
            state[:, self.block_ids] = torch.matmul(CNOT, state[:, self.block_ids])
        
        return state
                
    
class Entangle_layer(nn.Module):
    """A layer applying the 1/sqrt(2) XX+iZZ gate to all given pairs of qubits"""
    def __init__(self, qubit_pairs):
        """
        qubit_pairs: a list of tuples containing pairs of qubits to be entangled in a single layer
        """
        
        super().__init__()
        
        assert np.any([i!=j for (i,j) in qubit_pairs]), "Invalid pairs included"
        self.qubit_pairs = qubit_pairs
        
        
        X = torch.Tensor([[0,1],[1,0]]) / 2**0.25
        Z = (1j**0.5)*torch.Tensor([[1,0],[0,-1]]) / 2**0.25
        self.U = torch.stack((X,Z)).reshape(1,1,2,2,2).cdouble()
        
    def forward(self, state):
        indices = []
        state = state.repeat(1,1,2,1,1)
        reps = state.shape[2] // 2
        U = self.U.repeat_interleave(reps, dim=2)
        num_repeat_indices = 1
        for i,j in self.qubit_pairs:
            indices.append(i)
            indices.append(j)
            unique, count = np.unique(indices, return_counts=True)
            if count.max() > num_repeat_indices:
                state = state.repeat(1,1,2,1,1)
                U = U.repeat_interleave(2, dim=2)
                num_repeat_indices += 1
                    
            state[:,i] = torch.matmul(U, state[:,i])
            state[:,j] = torch.matmul(U, state[:,j])
            
        return state
  
    
    
    
    
    
    
    