# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""

from model import BasicModel
import torch.nn as nn
import torch

X = torch.Tensor([[0.1, 0.4], [0.6, 0.9]]).double()
Y = torch.Tensor([[-1, -1], [1, 1]]).double()
n_qubits = 2
criterion = nn.MSELoss()
model = BasicModel(n_qubits)
layers=[x.data for x in model.parameters()]
print(layers)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
losses = []
for epoch in range(2000):
    
    loss = None
    
    def loss_closure():
        x, y = X[0], Y[0]
        optimizer.zero_grad()
        state, O = model(x)
        
        loss = criterion(O.real, y.view(-1,1,1))
        
        if loss.requires_grad:
            loss.backward()
            
        return loss
    
    optimizer.step(loss_closure)
    layers += [x.data for x in model.parameters()]
    
    losses.append(loss_closure())
        