# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:27:00 2022

@author: Marius
"""

import torch
from torch.utils.data import Dataset
from numpy import pi

class DataFactory(Dataset):
    def __init__(self, batch_size: int, n_blocks: int, n_qubits: int):
        self.batch_size = batch_size 
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
            
    def next_batch(self):        
        X = torch.rand((self.batch_size,self.n_blocks,1,self.n_qubits,1))*2*pi
        Y = torch.rand((self.batch_size,self.n_blocks,1,1,1)).cdouble()*2 -1
        return X, Y
    
    