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
    def __init__(self, n_blocks: int, n_qubits: int, weights = None, weights_spread = [-np.pi/2,np.pi/2]):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1,1))
            nn.init.uniform_(self.weights, self.weights_spread[0], self.weights_spread[1])
            self.weights.cfloat()
        else:
            self.weights = weights
            if self.weights.shape[1] == n_blocks*n_qubits:
                self.weights = self.weights.view(-1, n_blocks, 1, n_qubits, 1, 1)
            else:
                shape = weights.shape
                self.weights = weights.view(shape[0], shape[1], 1, shape[2], 1, 1)
            if self.weights.shape[1] == 1 and self.n_blocks > 1:
                self.weights = self.weights.repeat(1,self.n_blocks,1, 1, 1, 1)
            if self.weights.shape[3] == 1 and self.n_qubits > 1:
                self.weights = self.weights.repeat(1,1,1, self.n_qubits, 1, 1)
                
            assert self.weights.shape[1] * self.weights.shape[3] == n_blocks*n_qubits, "Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count"
        

    def Rx(self, data=None):
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,1],[1,0]])
        if data == None:
            a = (self.weights/2).cos()
            b = (self.weights/2).sin()
        else:
            assert data.shape[1] == self.weights.nelement(), "Dimension of data incompatible"
            data = data.view((-1, *self.weights.shape[1:]))
            a = (self.weights*data/2).cos()
            b = (self.weights*data/2).sin()
        
        return a*identity - 1j*b*off_identity
        
           
    def forward(self, state, data=None):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n-qubit state (2**n x 1) 
        """
        Rxs = self.Rx(data).cfloat()
        U = torch.zeros(*Rxs.shape[:3], 2**self.n_qubits, 2**self.n_qubits).cfloat()
        for batch_idx in range(U.shape[0]):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rxs[batch_idx, block_idx, 0])
            
        state = torch.matmul(U, state) 

        return state
    
    
class Ry_layer(nn.Module):
    "A layer applying the Ry gate"
    def __init__(self, n_blocks: int, n_qubits: int, weights = None, weights_spread = [-np.pi/2,np.pi/2]):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1,1))
            nn.init.uniform_(self.weights, self.weights_spread[0], self.weights_spread[1])
            self.weights.cfloat()
        else:
            self.weights = weights
            if self.weights.shape[1] == n_blocks*n_qubits:
                self.weights = self.weights.view(-1, n_blocks, 1, n_qubits, 1, 1)
            else:
                shape = weights.shape
                self.weights = weights.view(shape[0], shape[1], 1, shape[2], 1, 1)
            if self.weights.shape[1] == 1 and self.n_blocks > 1:
                self.weights = self.weights.repeat(1,self.n_blocks,1, 1, 1, 1)
            if self.weights.shape[3] == 1 and self.n_qubits > 1:
                self.weights = self.weights.repeat(1,1,1, self.n_qubits, 1, 1)
                                
            assert self.weights.shape[1] * self.weights.shape[3] == n_blocks*n_qubits, "Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count"

    def Ry(self, data=None):
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,-1],[1,0]])
        if data == None:
            a = (self.weights/2).cos()
            b = (self.weights/2).sin()
        else:
            assert data.shape[1] == self.weights.nelement(), "Dimension of data incompatible"
            data = data.view((-1, *self.weights.shape[1:]))
            a = (self.weights*data/2).cos()
            b = (self.weights*data/2).sin()
        return a*identity + b*off_identity
        
           
    def forward(self, state, data=None):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n-qubit state (2**n x 1) 
        """
        
        Rys = self.Ry(data).cfloat()
        U = torch.zeros(*Rys.shape[:3], 2**self.n_qubits, 2**self.n_qubits).cfloat()
        for batch_idx in range(U.shape[0]):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rys[batch_idx, block_idx, 0])

        state = torch.matmul(U, state) 
        
        return state
        
    ### Consider how to include nograd if weights is given. This might be already incorporated?

class Rz_layer(nn.Module):
    "A layer applying the Rz gate"
    def __init__(self, n_blocks: int, n_qubits: int, weights = None, weights_spread = [-np.pi/2,np.pi/2]):
        """
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread

        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(1, self.n_blocks,1,self.n_qubits,1))
            nn.init.uniform_(self.weights, self.weights_spread[0], self.weights_spread[1])
            self.weights.cfloat()
        else:
            self.weights = weights
            if self.weights.shape[1] == n_blocks*n_qubits:
                self.weights = self.weights.view(-1, n_blocks, 1, n_qubits, 1)
            else:
                shape = weights.shape
                self.weights = weights.view(shape[0], shape[1], 1, shape[2], 1)
            if self.weights.shape[1] == 1 and self.n_blocks > 1:
                self.weights = self.weights.repeat(1,self.n_blocks,1, 1, 1)
            if self.weights.shape[3] == 1 and self.n_qubits > 1:
                self.weights = self.weights.repeat(1,1,1, self.n_qubits, 1)
                            
            assert self.weights.shape[1] * self.weights.shape[3] == n_blocks*n_qubits, "Dimensions of weight tensor are incompatable. Check the input has the right batch size, block and qubit count"

    def Rz(self):
        a = -1j*(self.weights/2)
        Z = torch.Tensor([[1, -1]])
        return (a*Z).exp() 
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x blocks x d&c x n_qubit state (2**n_qubits_perblock x 1) 
        """
        Rzs = self.Rz().cfloat()
        U = torch.zeros(*Rzs.shape[:3], 2**self.n_qubits, 1).cfloat()
        for batch_idx in range(U.shape[0]):
            for block_idx in range(self.n_blocks):
                U[batch_idx, block_idx, :] = krons(Rzs[batch_idx, block_idx, 0]).reshape(-1,1) #Reshape because using flattend matrix
            
        state = U*state 

        return state
    
class Hadamard_layer(nn.Module):
    def __init__(self, n_blocks: int, n_qubits: int):        
        super().__init__()
        
        self.H = torch.Tensor([[1, 1], [1, -1]]).reshape(1,2,2)/(2**0.5)
        self.H = krons(self.H.repeat(n_qubits,1,1)).cfloat()
        
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
            CNOT = self.gen_CNOT(pair).cfloat()
            state = torch.matmul(CNOT, state)
        
        return state
                
    
class Entangle_layer(nn.Module):
    """A layer applying the CNOT gate to given pairs of qubits in different blocks"""
    def __init__(self, coordinates, n_qubits):
        """
        qubit_pairs: a list of tuples containing pairs of qubits to be entangled in a single layer
        """
        
        super().__init__()
        
        self.n_qubits = n_qubits
        self.coordinates = coordinates
        # qubit pairs should be a list of tuples of the form ((block_idx1, qubit_idx1), (block_idx2, qubit_idx2))
        # the first tuple contains the control qubit coordinate, and the second tuple the target qubit coordinate
        
        self.I = torch.eye(2).cfloat()
        self.H = torch.Tensor([[1, 1], [1, -1]])/(2**0.5)
        self.H = self.H.cfloat()
        self.Z = torch.Tensor([[1,0],[0,-1]]).cfloat()
        self.S = torch.Tensor([[1,0],[0,1]]).cfloat()
        self.S[1,1] = -1j #This is actually S_conj just to save an unneccesary .conj() operation

        
    # Change this to feed in the explicit form of the matrix rather than calculating
    def control_U(self, Pauli):
        return torch.mm(self.S, Pauli)
    
    def target_U(self, Pauli):
        return torch.mm(self.H, torch.mm(self.S, torch.mm(Pauli, self.H)))
    
    def embed(self, q_idx: int, control_or_target):
        U = control_or_target
        if q_idx > 0 and q_idx < self.n_qubits - 1:
            pre_I = torch.eye(2**(q_idx))
            post_I = torch.eye(2**(self.n_qubits-q_idx-1))
            return torch.kron(pre_I, torch.kron(U, post_I))
        elif q_idx == 0:
            post_I = torch.eye(2**(self.n_qubits-1))
            return torch.kron(U, post_I)
        elif q_idx == self.n_qubits - 1 or q_idx == -1:
            pre_I = torch.eye(2**(self.n_qubits - 1))
            return torch.kron(pre_I, U)
        else:
            raise ValueError("Invalid qubit index given")    
    
    def forward(self, state):
        """
        state is a batch_size x n_blocks x d&c x 2**n_qubits x 1
        ent_pairs is a list of lists containing pairs of idxs of entangled blocks"""
        for gate in self.coordinates:
            state = state.repeat(1,1,2,1,1)
            
            for j, (block_idx, qubit_idx) in enumerate(gate):        
                    
                reps = state.shape[2] // 2   
                if j == 0:                 
                    state[:, block_idx, :reps] = torch.matmul(self.embed(qubit_idx, self.control_U(self.I)),
                                                              state[:, block_idx, :reps])
                    state[:, block_idx, reps:] = torch.matmul(self.embed(qubit_idx, self.control_U(self.Z)),
                                                              state[:, block_idx, reps:])
                if j == 1:                 
                    state[:, block_idx, :reps] = torch.matmul(self.embed(qubit_idx, self.target_U(self.I)),
                                                              state[:, block_idx, :reps])
                    state[:, block_idx, reps:] = torch.matmul(self.embed(qubit_idx, self.target_U(self.Z)),
                                                              state[:, block_idx, reps:])
                       
        return state
    
    
    
    
    
    
    
    