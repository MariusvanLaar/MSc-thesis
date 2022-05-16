# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi
import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-pi/2, pi/2))

import pickle
from sklearn.decomposition import PCA


# mnist = fetch_openml('mnist_784')
# # test_size: what proportion of original data is used for test set
# train_img, test_img, train_lbl, test_lbl = train_test_split(
#     mnist.data, mnist.target, test_size=1/7.0, random_state=0)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batches = [*range(1,6)]
for idx in batches:
    CIFAR = unpickle("cifar-10-batches-py/data_batch_"+str(idx))
    labels_ = np.array(CIFAR[b"labels"])
    idxs_0 = np.argwhere(labels_ == 0)
    idxs_1 = np.argwhere(labels_ == 8)
    try:
        X = np.concatenate((X, CIFAR[b"data"][idxs_0]))
        X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
        Y = np.concatenate((Y, labels_[idxs_0]))
        Y = np.concatenate((Y, labels_[idxs_1]))
    except NameError:
        X = CIFAR[b"data"][idxs_0]
        X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
        Y = labels_[idxs_0]
        Y = np.concatenate((Y, labels_[idxs_1]))
        
X = X[:,0]
print(X.shape)
scaler.fit(X)
data = scaler.transform(X)
pca = PCA()
pca.fit(data)
f = pca.transform(data)

print(data.shape)

def relabel(Y):
    #Y[Y==0] = 0
    Y[Y==8] = 1
    return Y

Y = relabel(Y)




cifar15 = {"data": data, "labels": Y}


pickling_on = open("CIFAR-PCA-08-train.pkl","wb")
pickle.dump(cifar15, pickling_on)
pickling_on.close()

# X, Y = None, None


CIFAR = unpickle("cifar-10-batches-py/test_batch")
labels_ = np.array(CIFAR[b"labels"])
idxs_0 = np.argwhere(labels_ == 0)
idxs_1 = np.argwhere(labels_ == 8)
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
print(X.shape)
#pca = PCA()
#pca.fit(X)
scaler.fit(X)
f = scaler.transform(X)
data = pca.transform(f)
print(data.shape)

def relabel(Y):
    #Y[Y==0] = 0
    Y[Y==8] = 1
    return Y

Y = relabel(Y)

cifar15 = {"data": data, "labels": Y}


pickling_on = open("CIFAR-PCA-08-test.pkl","wb")
pickle.dump(cifar15, pickling_on)
pickling_on.close()




