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



from models import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from datafactory import DataFactory
from torch.utils.data import DataLoader
from numpy import pi
import numpy as np
import time
from torch.utils.data import DataLoader
from scipy.stats import median_abs_deviation
from optimizers import SPSA



def train(model, optimizer, data_filename, batch_size, epochs, val_batch_size, n_blocks, n_qubits):
    criterion = nn.BCELoss()
    train_data = DataLoader(DataFactory(data_filename, test_train = "train"),
                            batch_size=batch_size, shuffle=True)
    test_data = DataLoader(DataFactory(data_filename, test_train = "test"),
                            batch_size=val_batch_size, shuffle=True)
    losses = []
    val_losses = []
    accs = []
    first_weight = []
    outputs = []
    for epoch in range(epochs):
        # if epoch == 0:
        #     for name, param in model.named_parameters():
        #         print(name, param)
        x, y = next(iter(train_data))
        y = y.float()
        loss = None
        
        def loss_closure():
            nonlocal loss
            
            optimizer.zero_grad()
            state, output = model(x)
            pred = output.reshape(*y.shape)
            #accs.append((torch.round(pred)==y).sum().item()/batch_size)
            loss = criterion(pred, y)
            #Backpropagation
            if loss.requires_grad:
                loss.backward()
            
            return loss

        optimizer.step(loss_closure)
                
        for name, param in model.named_parameters():
            first_weight.append(param.data.view(-1)[0].item())
        
        losses.append(loss.item())
        
        if epoch % 10 == 0 or epoch == (epochs-1):            
            
            x_val, y_val = next(iter(test_data))
            state, output = model(x_val)
            pred = output.reshape(*y_val.shape)
            y_val = y_val.float()
            outputs.append((y_val*pred)[0].item())
            accs.append((torch.round(pred)==y_val).sum().item()/val_batch_size)
            
            val_loss = criterion(pred, y_val)
            
            #print(epoch, val_loss.item())
            val_losses.append(val_loss.item())
    return losses, val_losses, accs, first_weight, outputs

def plot_mean_std_best(data, y_label, min_max, title):
    plt.figure()
    plt.ylabel(y_label)
    plt.plot(data.mean(axis=0), label="Mean", alpha=0.75)
    plt.fill_between(np.arange(len(data[0])), data.mean(axis=0)-data.std(axis=0), 
                     data.mean(axis=0)+data.std(axis=0), alpha=0.4)
    if min_max == 'min':
        best_idx = np.argmin(data.mean(axis=1))
    elif min_max == 'max':
        best_idx = np.argmax(data.mean(axis=1))
    plt.plot(data[best_idx], label="Best", alpha=0.75)
    plt.legend()    
    plt.title(title)
    plt.show()

lrs = [0.05]

for lr in lrs:
    print()
    print(lr)
    start = time.time()
    batch_size = 10
    n_blocks = 2
    n_qubits = 5
    epochs = 250
    reps = 5
    L = np.zeros((reps, epochs))
    #Lv = np.zeros((reps, epochs//10 + 1))
    accuracies = np.zeros((reps, epochs//10 + 1))
    for rep in range(reps):
        model = PQC_1A(n_blocks, n_qubits)
        #model = NeuralNetwork(n_blocks*n_qubits)
        optim = SPSA(model.parameters(), lr=lr)
        L_t, L_v, accs, FW, Os = train(model, optim, "wdbc", batch_size, epochs, batch_size, n_blocks, n_qubits)
        plt.plot(L_t)
        #plt.show()
        L[rep] = np.array(L_t)
        #Lv[rep] = np.array(L_v)
        accuracies[rep] = np.array(accs)
    plt.title(str(lr))
    plt.show()
    
    print(np.median(accuracies[:,-1]))

    
    # end = time.time()

    # print("Median validation loss at final timestep is:")
    # print(np.median(Lv[:,-1]))
    # print("Median absolute deviation:")
    # print(median_abs_deviation(Lv[:,-1]))
    # plot_mean_std_best(L, "Loss", "min", str(lr))
    # plot_mean_std_best(Lv, "Validation loss", "min", str(lr))
    # plot_mean_std_best(accuracies, "Accuracy", "max", str(lr))
    
    # print(end-start)







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
    
# n_qubits = 1
# n_blocks = 2
# batch_size = 1
# weights = torch.ones((batch_size, n_blocks, 1, n_qubits,1,1)).cdouble()*pi
# weights2 = torch.ones((batch_size, n_blocks, 1, n_qubits,1,1)).cdouble()*pi
# weights2[0,0] = 0
# model = TestModel(batch_size, n_blocks, n_qubits)
# s, O = model([weights, weights2])
# print(O)


# psi, O = model("")
# P = torch.zeros(16,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = krons(psi1[:,i])
#     P = torch.add(P, p)
# print(P)