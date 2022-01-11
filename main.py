# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


### TODO
# Batch functionality, works but is a bit clunky
# Datafactory



from model import BasicModel
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from datafactory import DataFactory
from torch.utils.data import DataLoader
from numpy import pi


def train(model, optimizer, batch_size, n_qubits):
    optim=optimizer
    criterion = nn.MSELoss()

    dataset = DataFactory(batch_size, n_qubits)
    losses = []
    for epoch in range(5):

        print(epoch)
        x, y = dataset.next_batch() 
        
        state, output = model(x)
        loss = criterion(output.real, y.real)
        
        #Backpropagation
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        losses.append(loss.item())
        print([p for p in model.parameters()])        
    return losses



n_qubits = 3
batch_size = 1
model = BasicModel(n_qubits, batch_size)
optim = torch.optim.AdamW(model.parameters(), lr=0.05)
L = train(model, optim, batch_size, n_qubits)

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