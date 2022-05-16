# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 09:40:30 2022

@author: Marius
"""

import pickle
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



filename = "wdbc.data"

data = np.zeros((569, 30))
labels = np.zeros((569))
f = open(filename, "r")
for j in range(569):
    line = f.readline()
    L  = line.split(",")[1:]
    if L[0] == "B":
        labels[j] = 0
    elif L[0] == "M":
        labels[j] = 1
    data[j] = np.array([float(x) for x in L[1:]])   
  


fname = "wdbc"

# synth_data = {"data": data, "labels": labels}
# pickling_on = open(fname+".pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()
