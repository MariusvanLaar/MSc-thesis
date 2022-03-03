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

        
        pickle_off = open(filename+test_train+".pkl", 'rb')
        data_dict = pickle.load(pickle_off)
        self.data = data_dict["data"]
        self.labels = data_dict["labels"]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label


if __name__ == "__main__":
    batch_size = 1
    train_data = DataLoader(DataFactory(test_train="train"),
                                batch_size=batch_size, shuffle=True)
    
    x, y = next(iter(train_data))
    if y == 0:
        lab = "dog"
    elif y == 1:
        lab = "automobile"
    
    # import matplotlib.pyplot as plt
    # plt.imshow(x.reshape(30,30))
    # plt.colorbar()
    # plt.title(lab)