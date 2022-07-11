# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:20:13 2022

@author: Marius
"""

import datasets, models
import models
import torch
import numpy as np
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt


def count_(data, threshold, N):
    data = [d for d in data if d < N*threshold]
    data = [d for d in data if d > -N*threshold]
    return len(data)


seed = 4554641
torch.manual_seed(seed)
np.random.seed(seed)
   
model_name = "PQC-4A"
n_qubits = 5
n_blocks = 2
n_layers = 2
obs = "First"
num_datapoints = 1000

counter = 0
X, Y = [], []
     
model = models.model_set[model_name](n_blocks, n_qubits, n_layers=n_layers, observable=obs)

outputs=[]
for j in range(1000):
    x = (torch.rand((1,n_qubits*n_blocks))*2 - 1)*np.pi
    O = model(x)
    outputs.append(O.item())
    
threshold = np.std(outputs)*1.001
print(threshold)
print(np.std(outputs))

outputs = []
while len(X) < num_datapoints:
    x = (torch.rand((1,n_qubits*n_blocks))*2 - 1)*np.pi

    O = model(x)
    outputs.append(O.item())
    if abs(O.item()) < threshold:
        if count_(Y, threshold, 1)/num_datapoints < 0.68:
            X.append(x.numpy()[0])
            Y.append(O.item())

    elif abs(O.item()) > threshold and abs(O.item()) < 2*threshold:
        if (count_(Y, threshold, 2) - count_(Y, threshold, 1)) / num_datapoints < (0.95-0.68):
            X.append(x.numpy()[0])
            Y.append(O.item())
            
    else:
        X.append(x.numpy()[0])
        Y.append(O.item())
        
    counter += 1
    if counter % num_datapoints == 0:
        print(str(counter)+" number of points tested")
        print(str(len(X))+" number of points eligible")
        plt.hist(outputs)
        plt.show()
        
print(np.std(outputs)**2)


# fname=f"datasets/data_files/{model_name}_{n_layers}_{n_blocks*n_qubits}_{obs}"
# synth_data = {"data": np.array(X), "labels": np.array(Y), "gen_seed": seed, 
#               "std": threshold}
# pickling_on = open(fname+".pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()

# import matplotlib.pyplot as plt
# Xp = [x for x, y in zip(X,Y) if y==1]
# Xn = [x for x, y in zip(X,Y) if y==0]
# Op = [o for o, y in zip(outputs,Y) if y==1]
# On = [o for o, y in zip(outputs,Y) if y==0]
    
# plt.scatter([x[0] for x in Xp], [x[1] for x in Xp], c="g")
# plt.scatter([x[0] for x in Xn], [x[1] for x in Xn], c="b", marker="x")
# plt.show()

