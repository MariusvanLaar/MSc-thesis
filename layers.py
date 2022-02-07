# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:32:13 2021

@author: Marius
"""

import torch
import torch.nn as nn
import numpy as np
from functools import reduce
import operator

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
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights = None, weights_spread = 0):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.batch_size = batch_size   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1,1))
            nn.init.uniform_(self.weights, -self.weights_spread, self.weights_spread)
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
        for batch_idx in range(U.shape[0]):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rxs[batch_idx, block_idx, 0])
            
        state = torch.matmul(U, state) 

        return state
    
    
class Ry_layer(nn.Module):
    "A layer applying the Ry gate"
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights = None, weights_spread = 0):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.batch_size = batch_size   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1,1))
            nn.init.uniform_(self.weights, -self.weights_spread, self.weights_spread)
            self.weights.cdouble()
        else:
            if weights.shape[0] == batch_size and weights.shape[1] == n_blocks and weights.shape[3] == n_qubits:
                self.weights = weights.view(batch_size, n_blocks, 1, n_qubits, 1, 1)     
            elif weights.shape[3] == 1 and self.n_qubits > 1: 
                self.weights = weights.repeat(1,1,1,self.n_qubits,1)
            else:
                raise RuntimeError("Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count")
        

    def Ry(self):
        a = (self.weights/2).cos()
        b = (self.weights/2).sin()
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,-1],[1,0]])
        return a*identity + b*off_identity
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n-qubit state (2**n x 1) 
        """
        
        Rys = self.Ry().cdouble()
        U = torch.zeros(*Rys.shape[:3], 2**self.n_qubits, 2**self.n_qubits).cdouble()
        for batch_idx in range(U.shape[0]):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rys[batch_idx, block_idx, 0])
            
        state = torch.matmul(U, state) 

        return state
        
    ### Consider how to include nograd if weights is given. This might be already incorporated?

class Rz_layer(nn.Module):
    "A layer applying the Rz gate"
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int, weights = None, weights_spread = 0):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.batch_size = batch_size   
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1))
            nn.init.uniform_(self.weights, -self.weights_spread, self.weights_spread)
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
        for batch_idx in range(U.shape[0]):
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
    def __init__(self, n_blocks: int, block_size: int, qubit_pairs: list):        
        super().__init__()
        
        self.n_blocks = n_blocks
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
            state = torch.matmul(CNOT, state)
        
        return state
                
    
class Entangle_layer(nn.Module):
    """A layer applying the 1/sqrt(2) XX+iZZ gate to given pairs of qubits"""
    def __init__(self, coordinates, n_qubits):
        """
        qubit_pairs: a list of tuples containing pairs of qubits to be entangled in a single layer
        """
        
        super().__init__()
        
        self.n_qubits = n_qubits
        self.coordinates = coordinates
        # qubit pairs should be a list of tuples of the form ((block_idx1, qubit_idx1), (block_idx2, qubit_idx2))
        # the first tuple contains the control qubit coordinate, and the second tuple the target qubit coordinate
        
        self.I = torch.eye(2).cdouble()
        self.H = torch.Tensor([[1, 1], [1, -1]])/(2**0.5)
        self.H = self.H.cdouble()
        self.Z = torch.Tensor([[1,0],[0,-1]]).cdouble()
        self.S = torch.Tensor([[1,0],[0,1]]).cdouble()
        self.S[1,1] = -1j #This is actually S_conj just to save an unneccesary .conj() operation

        
    def control_U(self, Pauli):
        return torch.mm(self.S, Pauli)
    
    def target_U(self, Pauli):
        return torch.mm(self.H, torch.mm(self.S, torch.mm(Pauli, self.H)))
    
    def embed(self, q_idx: int, control_or_target):
        U = control_or_target
        if q_idx > 0 and q_idx < self.n_qubits - 1:
            pre_I = torch.eye(2**(q_idx))
            post_I = torch.eye(2**(self.n_qubits-q_idx))
            return torch.kron(pre_I, torch.kron(U, post_I))
        elif q_idx == 0:
            post_I = torch.eye(2**(self.n_qubits-1))
            return torch.kron(U, post_I)
        elif q_idx == self.n_qubits - 1:
            pre_I = torch.eye(2**(q_idx))
            return torch.kron(pre_I, U)
        else:
            raise ValueError("Invalid qubit index given")    
    
    def forward(self, state, ent_pairs):
        """
        state is a batch_size x n_blocks x d&c x 2**n_qubits x 1
        ent_pairs is a list of lists containing pairs of idxs of entangled blocks"""
        # try:
        #     block_indices = reduce(operator.add, ent_pairs)
        #     num_repeat_indices = np.unique(block_indices, return_counts=True)[1].max()
        # except TypeError:
        #     block_indices = []
        #     num_repeat_indices = 0
        state = state.repeat(1,1,2,1,1)
        for gate in self.coordinates:
            # ent_pairs.append([gate[0][0], gate[1][0]])
            
            for j, (block_idx, qubit_idx) in enumerate(gate):
                # block_indices.append(block_idx)                
                # _, count = np.unique(block_indices, return_counts=True)
                # if count.max() > num_repeat_indices:
                #     state = state.repeat(1,1,2,1,1)
                #     num_repeat_indices += 1
                
                    
                reps = state.shape[2] // 2   
                if j == 0:                 
                    state[:, block_idx, :reps] = torch.matmul(self.embed(qubit_idx, self.control_U(self.I)), state[:, block_idx, :reps])
                    state[:, block_idx, reps:] = torch.matmul(self.embed(qubit_idx, self.control_U(self.Z)), state[:, block_idx, reps:])
                if j == 1:                 
                    state[:, block_idx, :reps] = torch.matmul(self.embed(qubit_idx, self.target_U(self.I)), state[:, block_idx, :reps])
                    state[:, block_idx, reps:] = torch.matmul(self.embed(qubit_idx, self.target_U(self.Z)), state[:, block_idx, reps:])
                       
        return state, 0#, ent_pairs
    
    
    
    ### OLD
class _Entangle_layer(nn.Module):
    """A layer applying the 1/sqrt(2) XX+iZZ gate to given pairs of qubits"""
    def __init__(self, coordinates, n_qubits):
        """
        qubit_pairs: a list of tuples containing pairs of qubits to be entangled in a single layer
        """
        
        super().__init__()
        
        self.n_qubits = n_qubits
        self.coordinates = coordinates
        self.ent_blocks = []
        # qubit pairs should be a list of tuples of the form ((block_idx1, qubit_idx1), (block_idx2, qubit_idx2))
        
        X = torch.Tensor([[0,1],[1,0]]) / 2**0.25
        Z = (1j**0.5)*torch.Tensor([[1,0],[0,-1]]) / 2**0.25
        
        if self.n_qubits > 1:       
            Uxs = torch.zeros((n_qubits, 2**n_qubits, 2**n_qubits))
            Uzs = torch.zeros((n_qubits, 2**n_qubits, 2**n_qubits))
            Uxs[0] = torch.kron(X, torch.eye(2**(n_qubits-1)))
            Uzs[0] = torch.kron(Z, torch.eye(2**(n_qubits-1)))
            Uxs[-1] = torch.kron(torch.eye(2**(n_qubits-1)), X)
            Uzs[-1] = torch.kron(torch.eye(2**(n_qubits-1)), Z)
            for i in range(1, n_qubits-1):
                Uxs[i] = torch.kron(torch.eye(2**i), torch.kron(X, torch.eye(2**(n_qubits-i-1))))
                Uxs[i] = torch.kron(torch.eye(2**i), torch.kron(Z, torch.eye(2**(n_qubits-i-1))))
            self.Uxs = Uxs.cdouble()
            self.Uzs = Uzs.cdouble()
        
        else:
            self.Uxs = X.view(1,2,2).cdouble()
            self.Uzs = Z.view(1,2,2).cdouble()
        
    def forward(self, state, ent_pairs):
        """
        state is a batch_size x n_blocks x d&c x 2**n_qubits x 1
        ent_pairs is a list of lists containing pairs of idxs of entangled blocks"""
        try:
            block_indices = reduce(operator.add, ent_pairs)
            num_repeat_indices = np.unique(block_indices, return_counts=True)[1].max()
        except TypeError:
            block_indices = []
            num_repeat_indices = 0
                
        for gate in self.coordinates:
            ent_pairs.append([gate[0][0], gate[1][0]])
            
            for block_idx, qubit_idx in gate:
                block_indices.append(block_idx)                
                _, count = np.unique(block_indices, return_counts=True)
                if count.max() > num_repeat_indices:
                    state = state.repeat(1,1,2,1,1)
                    num_repeat_indices += 1
                    
                reps = state.shape[2] // 2                    
                state[:, block_idx, :reps] = torch.matmul(self.Uxs[qubit_idx], state[:, block_idx, :reps])
                state[:, block_idx, reps:] = torch.matmul(self.Uzs[qubit_idx], state[:, block_idx, reps:])            
                       
        return state, ent_pairs
  
    
    
    
    
    
    
    