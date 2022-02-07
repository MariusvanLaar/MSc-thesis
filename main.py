# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


### TODO
# Nice way to create the circuit model, perhaps use inheritance with a base class?
# Dataset
# Timing
# Model can learn
# Data reuploading



from model import BasicModel, TestModel
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from datafactory import DataFactory
from torch.utils.data import DataLoader
from numpy import pi
import time
from torch.utils.data import DataLoader



def train(model, optimizer, batch_size, val_batch_size, n_blocks, n_qubits):
    optim=optimizer
    criterion = nn.BCELoss()
    train_data = DataLoader(DataFactory(test_train = "train"),
                            batch_size=batch_size, shuffle=True)
    test_data = DataLoader(DataFactory(test_train = "test"),
                            batch_size=val_batch_size, shuffle=True)
    losses = []
    val_losses = []
    for epoch in range(301):
        
        x, y = next(iter(train_data))
        
        state, output = model(x)
        output = torch.prod(output.real, dim=1)
        pred = 0.5*(output.reshape(*y.shape) + 1)
        y = y.double()
        loss = criterion(pred, y)
        
        #Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            val_model = BasicModel(val_batch_size, n_blocks, n_qubits)
            
            val_model.load_state_dict(model.state_dict())
            
            x_val, y_val = next(iter(test_data))
            state, output = val_model(x_val)
            output = torch.prod(output.real, dim=1)
            pred = 0.5*(output.reshape(*y_val.shape) + 1)
            y_val = y_val.double()
            val_loss = criterion(pred, y_val)
            
            print(epoch, val_loss.item())
            val_losses.append(val_loss.item())
    return losses, val_losses

# lrs = [0.001] #, 0.01, 0.05, 0.1, 0.5, 1]

# for lr in lrs:
#     print()
#     print(lr)
#     start = time.time()
#     batch_size = 4
#     n_blocks = 180
#     n_qubits = 5
#     model = BasicModel(batch_size, n_blocks, n_qubits)
#     optim = torch.optim.Adam(model.parameters(), lr=lr)
#     L_t, L_v = train(model, optim, batch_size, 20, n_blocks, n_qubits)
    
#     end = time.time()
#     plt.plot(L_t)
#     plt.show()
#     plt.plot(L_v)
#     plt.show()
    
#     print(end-start)







# times = []
# QUBITS = [*range(2, 7)]
# for q in QUBITS:
#     start = time.time()
#     n_qubits = 5
#     batch_size = 2 
#     n_blocks = q
#     model = BasicModel(batch_size, n_blocks, n_qubits)
#     optim = torch.optim.Adam(model.parameters(), lr=0.05)
#     L = train(model, optim, batch_size, n_blocks, n_qubits)
#     end = time.time()
#     times.append(end-start)
    
# plt.plot(QUBITS, times)
# plt.xlabel("Batchsize")
# plt.ylabel("Time, s")
# plt.title("")
#plt.savefig('QubitsTiming')
    
n_qubits = 1
n_blocks = 2
batch_size = 1
weights = torch.ones((batch_size, n_blocks, 1, n_qubits,1,1)).cdouble()*pi
weights2 = torch.ones((batch_size, n_blocks, 1, n_qubits,1,1)).cdouble()*pi
weights2[0,0] = 0
model = TestModel(batch_size, n_blocks, n_qubits)
s, O = model([weights, weights2])
print(O)


# psi, O = model("")
# P = torch.zeros(16,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = krons(psi1[:,i])
#     P = torch.add(P, p)
# print(P)