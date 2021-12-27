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
    def __init__(self, qubits, weights = None):
        """
        qubits: a list with the index of every qubit this layer is to be applied to
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.qubits = qubits
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(len(qubits)))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
        else:
            if len(weights) == 1 and len(qubits) > 1:
                self.weights = weights.repeat(len(qubits))
            elif len(weights) == len(qubits):
                self.weights = weights
            else:
                print("Dimensions of weight tensor are incompatable. Should be 1 or equal to len(qubits)")

    def Rx(self):
        L = len(self.weights)
        a = (self.weights/2).cos().view(-1,1,1)
        b = (self.weights/2).sin().view(-1,1,1)
        identity = torch.eye(2).reshape(1,2,2,).repeat(L,1,1,)
        off_identity = torch.Tensor([[0,1],[1,0]]).reshape(1,2,2,).repeat(L,1,1,)
        return a*identity - 1j*b*off_identity
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension qubits x d&c x density matrix (2 x 2) 
        Yet to include d&c dimension into this functions calculations
        When you do make sure to adjust .transpose as necessary
        """
        U = self.Rx().cdouble()
        state[self.qubits] = torch.bmm(U, state[self.qubits])
        return state
        
    ### Consider how to include nograd if weights is given. This might be already incorporated?
    
class Rz_layer(nn.Module):
    "A layer applying the Rz gate"
    def __init__(self, qubits, weights = None):
        """
        qubits: a list with the index of every qubit this layer is to be applied to
        weights: a tensor of rotation angles, if given from input data
        """
        
        super().__init__()
        
        self.qubits = qubits
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(len(qubits)))
            nn.init.uniform_(self.weights, 0, 2*np.pi)
        else:
            if len(weights) == 1 and len(qubits) > 1:
                self.weights = weights.repeat(len(qubits))
            elif len(weights) == len(qubits):
                self.weights = weights
            else:
                print("Dimensions of weight tensor are incompatable. Should be 1 or equal to len(qubits)")

    def Rz(self):
        L = len(self.weights)
        a = 1j*(self.weights/2).view(-1,1)
        Z = torch.Tensor([[-1,1]]).reshape(1,2,).repeat(L,1,)
        return torch.diag_embed((a*Z).exp(), dim1=1, dim2=2)
        
           
    def forward(self, state):
        """
        Take state to be a tensor with dimension qubits x d&c x density matrix (2 x 2) 
        Yet to include d&c dimension into this functions calculations
        When you do make sure to adjust .transpose as necessary
        """
        U = self.Rz().cdouble()
        state[self.qubits] = torch.bmm(U, state[self.qubits])
        return state