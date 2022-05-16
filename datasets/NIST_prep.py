# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 17:07:52 2022

@author: Marius
"""

import pickle
import numpy as np



filepath = "optdigits.tes"
file = open(filepath, "r")

len_tes = 1797
len_tra = 3823
data = np.zeros((len_tes+len_tra, 64))
labels = np.zeros((len_tes+len_tra))

for j in range(len_tes):
    line = file.readline()
    L  = line.split(",")
    data[j] = np.array([float(x) for x in L[:-1]])   
    labels[j] = int(L[-1])
        
filepath = "optdigits.tra"
file = open(filepath, "r")

for j in range(len_tra):
    line = file.readline()
    L  = line.split(",")
    data[j+len_tes] = np.array([float(x) for x in L[:-1]])   
    labels[j+len_tes] = int(L[-1])
        
# print(np.unique(labels, return_counts=True))
        
print(np.std(data, axis=0))
# data = np.delete(data, 1, 1) #Remove 2nd feature with no variance
# print(np.std(data, axis=0))

fname = "mnist"

synth_data = {"data": data, "labels": labels}
pickling_on = open(fname+".pkl","wb")
pickle.dump(synth_data, pickling_on)
pickling_on.close()