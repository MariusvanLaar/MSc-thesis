# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 21:32:13 2021

@author: Marius
"""

import torch
import torch.nn as nn


class Rx(nn.Module):
    "A layer applying the Rx gate"
    def __init__(self, qubits, weights = None):
        """
        qubits: a list with the index of every qubit this layer is to be applied to
        weights: a tensor of rotation angles from input data
        """
        
        super().__init__()
        
        self.qubits = qubits
        if weights is None:
            weights = nn.Parameter(torch.Tensor(len(qubits)))
            nn.init.uniform_(weights, 0, 2*np.pi)
            
    def forward(self, state):
        states = state[*qubits]
        
    ### Add feature that allows for a single parameter/angle to be input which is then applied to all given qubits
    ### Consider how to include nograd if weights is given.