# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:27:00 2022

@author: Marius
"""

import torch
from torch.utils.data import Dataset
from numpy import pi
import pickle
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader



class DataFactory(Dataset):
    def __init__(self, filename: str, test_train: str = "train"):
        
        super().__init__()   

        
        pickle_off = open("data/"+filename+"-"+test_train+".pkl", 'rb')
        data_dict = pickle.load(pickle_off)
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label

