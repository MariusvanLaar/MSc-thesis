# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


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
from optimizers import SPSA, CMA
import pandas as pd



def train(model, optimizer, data_filename, batch_size, epochs, val_batch_size, n_blocks, n_qubits, seed=5):
    criterion = nn.BCELoss()
    train_data = DataLoader(DataFactory(data_filename, test_train = "train"),
                            batch_size=batch_size, shuffle=True)
    test_data_ = DataFactory(data_filename, test_train = "test")
    test_data = DataLoader(test_data_,
                            batch_size=val_batch_size, shuffle=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    losses = []
    val_losses = []
    accs = []
    training_acc = []
    first_weight = []
    outputs = []
    for epoch in range(epochs):
        # if epoch == 0:
        #     for name, param in model.named_parameters():
        #         print(name, param)
        x, y = next(iter(train_data))
        y = y.float()
        loss = None
        train_acc = None
        def loss_closure():
            nonlocal loss
            nonlocal train_acc
            
            optimizer.zero_grad()
            
            state, output = model(x)
            pred = output.reshape(*y.shape)
            train_acc = (torch.round(pred)==y).sum().item()/batch_size
            loss = criterion(pred, y)
            #Backpropagation
            if loss.requires_grad:
                loss.backward()
            
            return loss

        optimizer.step(loss_closure)
        
        training_acc.append(train_acc)
                
        for name, param in model.named_parameters():
            first_weight.append(param.data.view(-1)[0].item())
            break
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:            
            
            x_val, y_val = next(iter(test_data))
            state, output = model(x_val)
            pred = output.reshape(*y_val.shape)
            y_val = y_val.float()
            outputs.append((y_val*pred)[0].item())
            accs.append((torch.round(pred)==y_val).sum().item()/val_batch_size)
            try:
                val_loss = criterion(pred, y_val)
            except RuntimeError:
                print("Validation step")
                idx_l = np.argwhere(pred<0)
                idx_u = np.argwhere(pred>1)
                print(x_val[idx_l].flatten()[:10])
                print(x_val[idx_u].flatten()[:10])
                s1, _ = model(x_val[idx_l], verbose=True)
                s2, __ = model(x_val[idx_u], verbose=True)
                print(pred[idx_l])
                print(pred[idx_u]%1)
            
            #print(epoch, val_loss.item())
            val_losses.append(val_loss.item())
            
    n_final_samples = min(len(test_data_), 500)
    final_data = DataLoader(test_data_, batch_size=n_final_samples, shuffle=True)
    x_val, y_val = next(iter(final_data))
    state, output = model(x_val)
    pred = output.reshape(*y_val.shape)
    y_val = y_val.float()
    try:
        val_loss = criterion(pred, y_val)
    except RuntimeError:
        print("Validation step")
        idx_l = np.argwhere(pred<0)
        idx_u = np.argwhere(pred>1)
        print(x_val[idx_l].flatten()[:10])
        print(x_val[idx_u].flatten()[:10])
        s1, _ = model(x_val[idx_l], verbose=True)
        s2, __ = model(x_val[idx_u], verbose=True)
        print(pred[idx_l])
        print(pred[idx_u]%1)
    
    val_losses.append(val_loss.item())
    accs.append((torch.round(pred)==y_val).sum().item()/n_final_samples)
    return losses, val_losses, accs, training_acc, first_weight, outputs

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
    
def moving_average(a, n=5) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


batch_size = 20
n_blocks = 2
n_qubits = 5
n_layers = 5
epochs = 400
reps = 1
lrs = [0.01, 0.05, 0.1]
results = []
for nlay in range(5, 51, 5):
    print(nlay)
    for lr in lrs:
        print()
        print(lr)
        start = time.time()
    
        L = np.zeros((reps, epochs))
        Lv = np.zeros((reps, epochs//10 + 1))
        accuracies = np.zeros((reps, epochs//10 + 1))
        for rep in range(reps):
            seed = int(time.time())%100000
            print(seed)
            
            model = PQC_4A(n_blocks, n_qubits, nlay)
            #model = NeuralNetwork(n_blocks*n_qubits)
            optim = torch.optim.Adam(model.parameters(), lr=lr)
            L_t, L_v, accs, t_acc, FW, Os = train(model, optim, "wdbc", batch_size, epochs, batch_size*5, n_blocks, n_qubits, seed)
            plt.plot(L_t)
            plt.ylabel("Training Loss")
            plt.xlabel("Epoch")
            plt.show()
            # plt.plot(t_acc)
            # plt.show()
            L[rep] = np.array(L_t)
            Lv[rep] = np.array(L_v)
            accuracies[rep] = np.array(accs)
            #fw = np.array(FW)
                    
        print(accuracies[:,-1])
    
        results.append({"nlayers":nlay, "lr":lr, "L_train":L, "L_val":Lv, "Acc":accuracies})
        # end = time.time()
        # print(end-start)



# psi, O = model("")
# P = torch.zeros(16,1)
# psi1 = psi[0]
# for i in range(psi1.shape[1]):
#     p = krons(psi1[:,i])
#     P = torch.add(P, p)
# print(P)