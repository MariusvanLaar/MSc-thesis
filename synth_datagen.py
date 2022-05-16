# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:20:13 2022

@author: Marius
"""

import datasets, models
from models.model import *
import torch
import numpy as np
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt


seed = 4554641
torch.manual_seed(seed)
np.random.seed(seed)
   
n_qubits = 5
n_blocks = 2
n_layers = 1
num_datapoints = 1000

num_points_found_pos = 0
num_points_found_neg = 0
counter = 0
X, Y = [], []
     
model = PQC_4AA(n_blocks, n_qubits, n_layers=n_layers)

outputs=[]
for j in range(1000):
    x = (torch.rand((1,n_qubits*n_blocks))*2 - 1)*np.pi
    O = model(x)
    outputs.append(O.item())
    
threshold = np.std(outputs)*1.001
print(threshold)

outputs = []
while num_points_found_pos != num_datapoints//2 or num_points_found_neg != num_datapoints//2:
    x = (torch.rand((1,n_qubits*n_blocks))*2 - 1)*np.pi

    O = model(x)
    outputs.append(O.item())
    if O.item() > 0.5 + threshold:
        if num_points_found_pos < num_datapoints//2:
            X.append(x.numpy()[0])
            Y.append(1)
            num_points_found_pos += 1
        
    elif O.item() < 0.5 - threshold:
        if num_points_found_neg < num_datapoints//2:
            X.append(x.numpy()[0])
            Y.append(0)
            num_points_found_neg += 1
        
    counter += 1
    if counter % num_datapoints == 0:
        print(str(counter)+" number of points tested")
        print(str(num_points_found_pos+num_points_found_neg)+" number of points eligible")
        plt.hist(outputs)
        plt.show()
        
print(np.std(outputs)**2)


fname=f"datasets/data_files/PQC4AA_{n_layers}_{n_blocks*n_qubits}"
synth_data = {"data": np.array(X), "labels": np.array(Y), "seed": seed}
pickling_on = open(fname+".pkl","wb")
pickle.dump(synth_data, pickling_on)
pickling_on.close()

import matplotlib.pyplot as plt
Xp = [x for x, y in zip(X,Y) if y==1]
Xn = [x for x, y in zip(X,Y) if y==0]
Op = [o for o, y in zip(outputs,Y) if y==1]
On = [o for o, y in zip(outputs,Y) if y==0]
    
plt.scatter([x[0] for x in Xp], [x[1] for x in Xp], c="g")
plt.scatter([x[0] for x in Xn], [x[1] for x in Xn], c="b", marker="x")
plt.show()

