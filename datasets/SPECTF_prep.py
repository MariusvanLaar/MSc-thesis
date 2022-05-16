# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:50:02 2022

@author: Marius
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import numpy as np
import matplotlib.pyplot as plt

filepath = "SPECTF.train"
file = open(filepath, "r")
train_data = np.loadtxt(file, delimiter=",")

filepath = "SPECTF.test"
file = open(filepath, "r")
test_data = np.loadtxt(file, delimiter=",")

class TanhE():
    """Tanh estimator"""
    def __init__(self, a=1):
        self.a = a
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        
    def transform(self, x):
        x -= self.mean
        x /= self.std
        return np.pi*0.5*np.tanh(self.a*x)
    
class Standard():
    """Standard Scaler"""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
        
    def transform(self, x):
        x -= self.mean
        x /= self.std
        return np.pi*x/2

tr_data = train_data[:,1:]
tr_labels = train_data[:,0]
te_data = test_data[:,1:]
te_labels = test_data[:,0]

# tr_data -= np.mean(tr_data, axis=0)
# te_data -= np.mean(tr_data, axis=0)

# tr_data /= np.std(tr_data, axis=0)
# te_data /= np.std(tr_data, axis=0)



# scaler_1 = TanhE(a=0.2)
# scaler_1.fit(tr_data)

# tr_data = scaler_1.transform(tr_data)
# te_data = scaler_1.transform(te_data)

# plt.hist(tr_data[:,5])
# plt.show()
# plt.hist(tr_data[:,8])
# plt.show()

# scaler_2 = MinMaxScaler((-np.pi/2, np.pi/2))
# scaler_2.fit(tr_data)

# tr_data = scaler_2.transform(tr_data)
# te_data = scaler_2.transform(te_data)
 
# print(np.unique(te_labels, return_counts=True))

fname = "spectf"

# synth_data = {"data": tr_data, "labels": tr_labels}
# pickling_on = open(fname+"-train.pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()

# synth_data = {"data": te_data, "labels": te_labels}
# pickling_on = open(fname+"-test.pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()

data = np.concatenate((tr_data, te_data))
labels = np.concatenate((tr_labels, te_labels))

scaler_1 = TanhE(a=0.3)
for col in range(2, 5):
    X = data[:,col]
    plt.hist(X)
    plt.show()
    scaler_1.fit(X)
    X_1 = scaler_1.transform(X)
    plt.hist(X_1)
    plt.show()


# synth_data = {"data": data, "labels": labels}
# pickling_on = open(fname+".pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()

