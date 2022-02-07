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

Z = np.array([[1,0],[0,-1]])
#X = np.array([[0,1],[1,0]])
I = torch.eye(2)

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-pi/2, pi/2))

import pickle

# data = pd.read_csv("../slice_localization_data/slice_localization_data.csv", nrows=1161)
# X = data[data.keys()[1:-1]]
# Y = data["reference"]
# scaler.fit(X)
# datat = scaler.transform(X)


# mnist = fetch_openml('mnist_784')
# # test_size: what proportion of original data is used for test set
# train_img, test_img, train_lbl, test_lbl = train_test_split(
#     mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict

# batches = [*range(1,6)]
# for idx in batches:
#     CIFAR = unpickle("../cifar-10-batches-py/data_batch_"+str(idx))
#     labels_ = np.array(CIFAR[b"labels"])
#     idxs_0 = np.argwhere(labels_ == 1)
#     idxs_1 = np.argwhere(labels_ == 5)
#     try:
#         X = np.concatenate((X, CIFAR[b"data"][idxs_0]))
#         X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
#         Y = np.concatenate((Y, labels_[idxs_0]))
#         Y = np.concatenate((Y, labels_[idxs_1]))
#     except NameError:
#         X = CIFAR[b"data"][idxs_0]
#         X = np.concatenate((X, CIFAR[b"data"][idxs_1]))
#         Y = labels_[idxs_0]
#         Y = np.concatenate((Y, labels_[idxs_1]))
        
# X = X[:,0]
# grey = (X[:,:1024] + X[:, 1024:2048] + X[:, 2048:] ) / 3

# grey = grey.reshape((-1,32,32))[:,1:-1, 1:-1].reshape((-1,900))
# print(grey.shape)
# scaler.fit(grey)
# data = scaler.transform(grey)
# print(data.shape)

# def relabel(Y):
#     Y[Y==5] = 0
#     #Y[Y==1] = 1
#     return Y

# Y = relabel(Y)




# cifar15 = {"data": data, "labels": Y}


# pickling_on = open("CIFAR-15-train.pkl","wb")
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
# grey = (X[:,:1024] + X[:, 1024:2048] + X[:, 2048:] ) / 3

# grey = grey.reshape((-1,32,32))[:,1:-1, 1:-1].reshape((-1,900))
# print(grey.shape)
# scaler.fit(grey)
# data = scaler.transform(grey)
# print(data.shape)

# def relabel(Y):
#     Y[Y==5] = 0
#     #Y[Y==1] = 1
#     return Y

# Y = relabel(Y)




# cifar15 = {"data": data, "labels": Y}


# pickling_on = open("CIFAR-15-test.pkl","wb")
# pickle.dump(cifar15, pickling_on)
# pickling_on.close()

# from sklearn.decomposition import PCA

# scaler.fit(grey)
# grey = scaler.transform(grey)

# cifar_comps = []

# for r in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
#     # Make an instance of the Model
#     pca = PCA(r)

#     pca.fit(grey)

#     print(r, pca.n_components_)
#     cifar_comps.append(pca.n_components_)
    
# mnist_comps = []
    
# scaler.fit(train_img)

# # Apply transform to both the training set and the test set.
# train_img = scaler.transform(train_img)
# test_img = scaler.transform(test_img)

# for r in [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]:
#     # Make an instance of the Model
#     pca = PCA(r)

#     pca.fit(train_img)

#     print(r, pca.n_components_)
#     mnist_comps.append(pca.n_components_)
    
    
# print(pd.DataFrame({"Required explained variance": [0.5, 0.75, 0.9, 0.95, 0.99, 0.999],
#                     "# Principle components CIFAR15": cifar_comps,
#                     "# Principle components MNIST": mnist_comps}).to_string())

I = np.identity(2, dtype=complex)
Z = I.copy()
Z[1,1] = -1
S = I.copy()
S[1,1] = -1j
H = np.array([[1,1],[1,-1]]) / np.sqrt(2)

IH = np.kron(I,H)
SSc = np.kron(S, S)
U = np.kron(I,I) - 1j*np.kron(Z,Z)
U /= np.sqrt(2)*(1-1j)

def ap(U1, U2):
    return np.matmul(U1, U2)

print(ap(IH, ap(SSc, ap(U, IH))).round(2))

print(ap(H, ap(S,H)))

Uc1 = S
Uc2 = ap(S, Z)

Ut1 = ap(H, ap(S, H))
Ut2 = ap(H, ap(S, ap(Z, H)))

T = (np.kron(Uc1, Ut1) - 1j*np.kron(Uc2, Ut2))/ ((2**0.5)*((-1j)**0.5))
print(ap(T, ap(T,T)))



