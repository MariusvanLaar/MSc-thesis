# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 18:10:43 2022

@author: Marius
"""

from .datafactory import DataFactory
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from torch.utils.data import Dataset



class WDBC(DataFactory):
    def __init__(self, n_features):
        self.permute = True
        super().__init__("wdbc", self.permute, n_features)
        self.scaler = MinMaxScaler((-np.pi/2, np.pi/2))
        self.mean = None
        self.data_info["loss"] = "BCE"

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        self.scaler.fit(X)
        
    def transform(self, X):
        X -= self.mean
        X = self.scaler.transform(X) 
        return X
    
   
class ION(DataFactory):
    def __init__(self, n_features):
        self.permute = True
        super().__init__("ion", self.permute, n_features)
        self.mean = 0
        self.pca = PCA()
        self.scaler = MinMaxScaler((-np.pi/2, np.pi/2))
        self.data_info["loss"] = "BCE"

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        X = self.pca.fit_transform(X)
        self.scaler.fit(X)
        
    def transform(self, X):
        X -= self.mean
        X = self.pca.transform(X)
        return self.scaler.transform(X)

class SONAR(DataFactory):
    """Consider what preprocessing to do here still"""
    def __init__(self, n_features):
        self.permute = True
        super().__init__("sonar", self.permute, n_features)
        self.scaler = MinMaxScaler((-np.pi/2, np.pi/2))
        self.mean = None
        self.data_info["loss"] = "BCE"

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        self.scaler.fit(X)
        
    def transform(self, X):
        X -= self.mean
        X = self.scaler.transform(X) 
        return X
    
class SPECTF(DataFactory):
    def __init__(self, n_features):
        self.permute = True
        super().__init__("spectf", self.permute, n_features)
        self.mean = 0
        self.pca = PCA()
        self.scaler = MinMaxScaler((-np.pi/2, np.pi/2))
        self.data_info["loss"] = "BCE"

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X -= self.mean
        X = self.pca.fit_transform(X)
        self.scaler.fit(X)
        
    def transform(self, X):
        X -= self.mean
        X = self.pca.transform(X)
        return self.scaler.transform(X)
    
class MNIST(DataFactory):
    """ Still to implement specific digit classes for binary classification """
    def __init__(self, n_features):
        self.permute = False
        super().__init__("mnist", self.permute, n_features)
        self.scaler = MinMaxScaler((0, np.pi))
        self.data_info["loss"] = "BCE"

    def fit(self, X):
        self.scaler.fit(X)
        
    def transform(self, X):
        X = self.scaler.transform(X) 
        return X
    
class synth_dataset(DataFactory):
    def __init__(self, model, n_features, layers):
        self.fname = f"{model}_{layers}_{n_features}"
        super().__init__(self.fname, False, n_features)
        self.data_info["loss"] = "MSE"
        self.data_info["return_probs"] = False
        
    def fit(self, X):
        pass
    def transform(self, X):
        return X

class SYNTH_4A(synth_dataset):
    def __init__(self, n_features):
        super().__init__("PQC4A", n_features, 4)
        
class SYNTH_4AA(synth_dataset):
    def __init__(self, n_features):
        super().__init__("PQC4AA", n_features, 4)
 
        
class SIMPLE(Dataset):
    def __init__(self, n_features):
        super().__init__()
        self.len_data = 500
        self.data = np.random.choice([-1,1], size=(self.len_data, n_features))*np.pi/2
        self.labels = (np.prod(self.data, axis=1)+1)*0.5
        self.data_info["loss"] = "BCE"
        
    def __len__(self):
        return self.len_data

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        return img, label

    def fit(self, X):
        pass
    def transform(self, X):
        return X
        
        
        
