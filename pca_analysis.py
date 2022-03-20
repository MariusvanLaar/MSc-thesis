# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#scaler = MinMaxScaler((-pi/2, pi/2))
scaler = StandardScaler()
import pickle
from sklearn.decomposition import PCA



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

for lab_1 in range(9):
    for lab_2 in range(lab_1+1, 10):
        X = np.array([])
        Y = np.array([])
        print(lab_1, lab_2)
        batches = [*range(1,6)]
        for idx in batches:
            CIFAR = unpickle("../cifar-10-batches-py/data_batch_"+str(idx))
            labels_ = np.array(CIFAR[b"labels"])
            idxs_0 = np.argwhere(labels_ == lab_1)
            idxs_1 = np.argwhere(labels_ == lab_2)
            if len(idxs_0) == 0:
                print("Label "+str(lab_1)+" not found")
            try:
                X = np.concatenate((X, CIFAR[b"data"][idxs_0]))
                X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
                Y = np.concatenate((Y, labels_[idxs_0]))
                Y = np.concatenate((Y, labels_[idxs_1]))
            except ValueError:
                X = CIFAR[b"data"][idxs_0]
                X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
                Y = labels_[idxs_0]
                Y = np.concatenate((Y, labels_[idxs_1]))
                
        X = X[:,0]
        scaler.fit(X)
        data = scaler.transform(X)
        
        exp_vars = [0.5, 0.75, 0.9, 0.95, 0.99]
        
        for e_v in exp_vars:
            pca = PCA(n_components=e_v)
            pca.fit(data)
            #f = pca.transform(data)
            print(e_v, pca.n_components_)

def relabel(Y):
    Y[Y==5] = 0
    #Y[Y==1] = 1
    return Y

Y = relabel(Y)





# cifar15 = {"data": data, "labels": Y}


# pickling_on = open("CIFAR-PCA-15-train.pkl","wb")
# pickle.dump(cifar15, pickling_on)
# pickling_on.close()

# X, Y = None, None


# CIFAR = unpickle("../cifar-10-batches-py/test_batch")
# labels_ = np.array(CIFAR[b"labels"])
# idxs_0 = np.argwhere(labels_ == 1)
# idxs_1 = np.argwhere(labels_ == 5)
# try:
#     X = np.concatenate((X, CIFAR[b"data"][idxs_0]))
#     X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
#     Y = np.concatenate((Y, labels_[idxs_0]))
#     Y = np.concatenate((Y, labels_[idxs_1]))
# except ValueError:
#     X = CIFAR[b"data"][idxs_0]
#     X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
#     Y = labels_[idxs_0]
#     Y = np.concatenate((Y, labels_[idxs_1]))
        
# X = X[:,0]
# print(X.shape)
# #pca = PCA()
# #pca.fit(X)
# f = pca.transform(X)
# scaler.fit(f)
# data = scaler.transform(f)
# print(data.shape)

# def relabel(Y):
#     Y[Y==5] = 0
#     #Y[Y==1] = 1
#     return Y

# Y = relabel(Y)

# cifar15 = {"data": data, "labels": Y}


# pickling_on = open("CIFAR-PCA-15-test.pkl","wb")
# pickle.dump(cifar15, pickling_on)
# pickling_on.close()




