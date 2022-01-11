# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:27:00 2022

@author: Marius
"""

import torch
from torch.utils.data import Dataset

class DataFactory(Dataset):
    def __init__(self, batch_size: int, input_dim: int):
        self.batch_size = batch_size 
        self.input_dim = input_dim
    
    def next_batch(self):
        X = torch.rand((self.batch_size, self.input_dim,1,1,1)).cdouble()*2 -1
        Y = torch.rand((self.batch_size, self.input_dim,1,1,1)).cdouble()*2 -1
        return X, Y
    
    