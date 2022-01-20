# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:27:00 2022

@author: Marius
"""

import torch
from torch.utils.data import Dataset
from numpy import pi

class DataFactory(Dataset):
    def __init__(self, batch_size: int, n_qubits: int, input_dim: int):
        self.batch_size = batch_size 
        self.n_qubits = n_qubits
        self.input_dim = input_dim
            
    def next_batch(self):        
        X = torch.rand((self.batch_size, self.input_dim,1,1,1))*2*pi
        Y = torch.rand((self.batch_size,1,1,1)).cdouble()*2 -1
        return X, Y
    
    