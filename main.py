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



from model import BasicModel
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

    dataset = DataFactory(batch_size, n_qubits)
    losses = []
    for epoch in range(4):
        x, y = dataset.next_batch() 
        
        state, output = model(x)
        loss = criterion(output.real, y.real)
        
        #Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        losses.append(loss.item())
    return losses


start = time.time()
n_qubits = 3
batch_size = 1
model = BasicModel(n_qubits, batch_size)
# psi, O = model("")
# P = torch.zeros(8,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = torch.kron(psi1[0,i], torch.kron(psi1[1,i], psi1[2,i]))
#     P = torch.add(P, p)
# print(P)
    
optim = torch.optim.AdamW(model.parameters(), lr=0.05)
# L = train(model, optim, batch_size, n_qubits)
# end = time.time()
# print(end-start)
# losses = []
# for i in range(10):
#     train_features, train_labels = next(iter(train_dataloader))
#     print(train_features, train_labels)
#     state, O = model(train_features)
#     print(state.shape)
#     loss = criterion(O.real, train_labels)
        
        
#     #Backpropagation
#     optim.zero_grad()
#     loss.backward(retain_graph=True)
#     optim.step()
        
#     losses.append(loss.item())


# plt.plot(losses, alpha=0.5)
#plt.hlines(1, 0, 500, colors="r")
#plt.ylim(0.9, 1.1)