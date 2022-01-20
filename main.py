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


def train(model, optimizer, batch_size, n_qubits):
    optim=optimizer
    criterion = nn.MSELoss()

    dataset = DataFactory(batch_size, n_qubits, 10)
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

times = []
QUBITS = [*range(5, 13)]
for q in QUBITS:
    start = time.time()
    n_qubits = q
    batch_size = 1
    model = BasicModel(n_qubits, batch_size)
    optim = torch.optim.AdamW(model.parameters(), lr=0.05)
    L = train(model, optim, batch_size, n_qubits)
    end = time.time()
    times.append(end-start)
    
plt.plot(QUBITS, times)
plt.xlabel("Qubits")
plt.ylabel("Time, s")
plt.title("")
# plt.savefig('DCdimTiming')
    
# n_qubits = 2
# batch_size = 1
# model = TestModel(n_qubits, batch_size,1)
# s, O = model(0)


# psi, O = model("")
# P = torch.zeros(16,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = krons(psi1[:,i])
#     P = torch.add(P, p)
# print(P)