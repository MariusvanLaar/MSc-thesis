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


def train(model, optimizer, batch_size, n_blocks, n_qubits):
    optim=optimizer
    criterion = nn.MSELoss()

    dataset = DataFactory(batch_size, n_blocks, n_qubits)
    losses = []
    for epoch in range(300):
        x, y = dataset.next_batch() 
        
        state, output = model(x)
        loss = criterion(output.real, y.real)
        
        #Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        losses.append(loss.item())
    return losses

# times = []
# QUBITS = [*range(1, 5)]
# for q in QUBITS:
#     start = time.time()
#     n_qubits = 6
#     batch_size = q
#     n_blocks = 2
#     model = BasicModel(batch_size, n_blocks, n_qubits)
#     optim = torch.optim.AdamW(model.parameters(), lr=0.05)
#     L = train(model, optim, batch_size, n_blocks, n_qubits)
#     end = time.time()
#     times.append(end-start)
    
# plt.plot(QUBITS, times)
# plt.xlabel("Batchsize")
# plt.ylabel("Time, s")
# plt.title("")
#plt.savefig('QubitsTiming')
    
n_qubits = 2
n_blocks = 2
batch_size = 1
weights = torch.ones((batch_size, n_blocks, 1, n_qubits,1,1)).cdouble()*pi
model = TestModel(batch_size, n_blocks, n_qubits)
s, O = model(weights)
print(s)


# psi, O = model("")
# P = torch.zeros(16,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = krons(psi1[:,i])
#     P = torch.add(P, p)
# print(P)