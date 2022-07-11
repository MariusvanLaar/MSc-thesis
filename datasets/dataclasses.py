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
        self.scaler = MinMaxScaler((-np.pi, np.pi))
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
    
class MNIST_(DataFactory):
    """ Still to implement specific digit classes for binary classification """
    def __init__(self, n_features):
        self.permute = False
        super().__init__("mnist", self.permute, n_features)
        self.scaler = MinMaxScaler((0, np.pi))
        self.data_info["loss"] = "BCE"
        if n_features == 64:
            self.data_info["obs_multiplier"] = (10**7)**(1/8)

    def fit(self, X):
        self.scaler.fit(X)
        
    def transform(self, X):
        X = self.scaler.transform(X) 
        return X
    
class MNIST_2digit_(MNIST_):
    """ Dataclass for two labels"""
    def __init__(self, n_features, d1, d2):
        super().__init__(n_features)
        self.data_1 = self.data[self.labels==d1]
        self.labels_1 = self.labels[self.labels==d1]
        self.data = np.concatenate((self.data_1, self.data[self.labels==d2]))
        self.labels = np.concatenate((self.labels_1, self.labels[self.labels==d2]))
        self.labels[self.labels==d1] = 0
        self.labels[self.labels==d2] = 1
        
class MNIST_23(MNIST_2digit_):
    """ Dataclass for 0 and 1 labels"""
    def __init__(self, n_features):
        super().__init__(n_features, 2, 3)
        
class MNIST_13(MNIST_2digit_):
    """ Dataclass for 0 and 1 labels"""
    def __init__(self, n_features):
        super().__init__(n_features, 1, 3)
        

    
class synth_pqc_dataset_(DataFactory):
    def __init__(self, model, n_features, layers, obs):
        self.fname = f"{model}_{layers}_{n_features}_{obs}"
        super().__init__(self.fname, False, n_features)
        self.data_info["loss"] = "MSE"
        self.data_info["return_probs"] = False
        
    def fit(self, X):
        pass
    def transform(self, X):
        return X

class SYNTH_4F(synth_pqc_dataset_):
    def __init__(self, n_features):
        super().__init__("PQC-4A", n_features, 2, "First")
        
class SYNTH_4F_random(synth_pqc_dataset_):
    def __init__(self, n_features):
        super().__init__("PQC-4A", n_features, 2, "First")
        self.labels = self.labels[torch.randperm(len(self.labels))]
        
class SYNTH_4A(synth_pqc_dataset_):
    def __init__(self, n_features):
        super().__init__("PQC-4A", n_features, 2, "All")
        
class ising_dataset_(DataFactory):
    def __init__(self, n_features, n_spins, obs):
        self.fname = f"TIsing_{n_spins}_{obs}"
        super().__init__(self.fname, False, n_features)
        self.data_info["loss"] = "MSE"
        self.data_info["return_probs"] = False
        self.data_info["X_holdout"] = self.data_info["X_holdout"][:,:n_features]
        
    def fit(self, X):
        pass
    def transform(self, X):
        return X
    
class ISING_10(ising_dataset_):
    def __init__(self, n_features):
        super().__init__(n_features, 10, 0)
 
    
class CIRCLE(DataFactory):
    def __init__(self, n_features):
        self.fname = f"Circle_{n_features}"
        super().__init__(self.fname, False, n_features)
        self.data_info["loss"] = "BCE"
        self.data_info["return_probs"] = True
        
    def fit(self, X):
        pass
    def transform(self, X):
        return X
        
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
        
        
        
