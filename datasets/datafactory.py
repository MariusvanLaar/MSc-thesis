# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:27:00 2022

@author: Marius
"""

import torch
from torch.utils.data import Dataset
import pickle
import os


class DataFactory(Dataset):
    def __init__(self, filename: str, permute, n_features):
        
        super().__init__()   
        pickle_off = open("datasets"+os.sep+"data_files"+os.sep+filename+".pkl", 'rb')
        data_dict = pickle.load(pickle_off)
        self.data = data_dict["data"][:,:n_features]
        if permute:
            self.data = self.data[:, torch.randperm(n_features)]
        self.labels = data_dict["labels"]
        self.data_info = dict((k,data_dict[k]) for k in data_dict.keys()
                              if k not in ["data", "labels"])
        self.data_info["return_probs"] = True #Whether to normalize model output to [0,1] range (if true)
        
        
    def __len__(self):
        return len(self.labels)
    
    def num_features(self):
        return self.data.shape[1]
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label
    
class FoldFactory(Dataset):
    def __init__(self, X, Y):
        
        super().__init__()   
        self.data = X
        self.labels = Y

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label
    

       
        
        
        
        
        
