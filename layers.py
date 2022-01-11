# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:32:13 2021

@author: Marius
"""

import torch
import torch.nn as nn
import numpy as np


class Rx_layer(nn.Module):
    "A layer applying the Rx gate"
    def __init__(self, qubits: list, batch_size: int, weights = None):
        """
        qubits: a list with the index of every qubit this layer is to be applied to
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.qubits = qubits
        self.num_qubits = len(qubits)
        self.batch_size = batch_size
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(self.batch_size, self.num_qubits,1,1,1))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
            self.weights.cdouble()
        else:
            if weights.shape[1] == 1 and self.num_qubits > 1:
                self.weights = weights.repeat(1, self.num_qubits)
            elif weights.shape[0] == batch_size and weights.shape[1] == len(qubits):
                self.weights = weights
            else:
                raise RuntimeError("Dimensions of weight tensor are incompatable. Check the input has the right batch size and qubit count")
        

    def Rx(self):
        a = (self.weights/2).cos()
        b = (self.weights/2).sin()
        identity = torch.eye(2)
        off_identity = torch.Tensor([[0,1],[1,0]])
        return a*identity - 1j*b*off_identity
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension batch x qubits x d&c x 1qubit state (2 x 1) 
        """
        U = self.Rx().cdouble()
        state[:,self.qubits] = torch.matmul(U, state[:,self.qubits]) 

        return state
        
    ### Consider how to include nograd if weights is given. This might be already incorporated?

class Rz_layer(nn.Module):
    "A layer applying the Rz gate"
    def __init__(self, qubits: list, batch_size: int, weights = None):
        """
        qubits: a list with the index of every qubit this layer is to be applied to
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.qubits = qubits
        self.num_qubits = len(qubits)
        self.batch_size = batch_size
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(self.batch_size, self.num_qubits))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
        else:
            if weights.shape[1] == 1 and len(qubits) > 1:
                self.weights = weights.repeat(1, len(qubits))
            elif weights.shape[0] == batch_size and weights.shape[1] == len(qubits):
                self.weights = weights
            else:
                raise RuntimeError("Dimensions of weight tensor are incompatable. Check the input has the right batch size and qubit count")

    def Rz(self):
        a = 1j*(self.weights/2)
        Z = torch.Tensor([[-1,1]]).reshape(1,1,2,).repeat(self.batch_size,self.num_qubits,1,)
        return torch.diag_embed((a*Z).exp(), dim1=2, dim2=3)
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension qubits x d&c x density matrix (2 x 2) 
        Yet to include d&c dimension into this functions calculations
        When you do make sure to adjust .transpose as necessary
        """
        U = self.Rz().cdouble()
        state[:,self.qubits] = torch.matmul(U, state[:,self.qubits])
        
        return state
    
class Entangle_layer(nn.Module):
    """A layer applying the 1/sqrt(2) XX+iZZ gate to all given pairs of qubits"""
    def __init__(self, qubit_pairs):
        """
        qubit_pairs: a list of tuples containing pairs of qubits to be entangled in a single layer
        """
        
        super().__init__()
        
        self.qubit_pairs = qubit_pairs
        
        X = torch.Tensor([[0,1],[1,0]]) / 2**0.25
        Z = (1j**0.5)*torch.Tensor([[1,0],[0,-1]]) / 2**0.25
        self.U = torch.stack((X,Z)).reshape(1,1,2,2,2).cdouble()
        
    def forward(self, state):
        state = state.repeat(1,1,2,1,1)
        if state.shape[2] != self.U.shape[2]:
            repeats = state.shape[2] // self.U.shape[2]
            U = self.U.repeat(1,1,repeats,1,1).cdouble()
        else:
            U = self.U.cdouble()
        for i,j in self.qubit_pairs:
            state[:,i] = torch.matmul(U, state[:,i])
            state[:,j] = torch.matmul(U, state[:,j])
            
        return state
  
    
    
    
    
    
    
    