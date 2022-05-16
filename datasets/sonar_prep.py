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



filename = "sonar.all-data"

data = np.zeros((208, 60))
labels = np.zeros((208))
f = open(filename, "r")
for j in range(208):
    line = f.readline()
    L  = line.split(",")
    if L[-1] == "R\n":
        labels[j] = 0
    elif L[-1] == "M\n":
        labels[j] = 1
    data[j] = np.array([float(x) for x in L[:-1]])   
  


fname = "sonar"
print(np.unique(labels, return_counts=True))
synth_data = {"data": data, "labels": labels}
pickling_on = open(fname+".pkl","wb")
pickle.dump(synth_data, pickling_on)
pickling_on.close()
