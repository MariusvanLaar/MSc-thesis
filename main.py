# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 10:10:55 2021

@author: Marius
"""


from models import *
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from datasets.datafactory import DataFactory, FoldFactory
import datasets
from torch.utils.data import DataLoader
from numpy import pi
import numpy as np
import time
from torch.utils.data import DataLoader
from scipy.stats import median_abs_deviation
from optimizers import SPSA, CMA
import pandas as pd
from sklearn.model_selection import KFold




def train(model, optim, data_filename, batch_size, epochs, val_batch_size, kfolds=10, seed=5, **kwargs):
    args = dict(model=model, dataset=data_filename, batch_size=batch_size, epochs=epochs, val_batch_size=val_batch_size, 
                kfolds=kfolds, kwargs=kwargs)
    
    torch.manual_seed(seed+123456789)
    np.random.seed(seed+123456789)

    criterion = nn.BCELoss()
    dataclass = datasets.all_datasets[data_filename]()
    kf = KFold(kfolds, shuffle=True, random_state=seed)
    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataclass.data)):
        torch.manual_seed(seed+fold)
        np.random.seed(seed+fold)
        
        if "index" in kwargs.keys():
            model = args["model"](kwargs["n_blocks"], kwargs["n_qubits"], index=kwargs["index"])
        else:
            model = args["model"](kwargs["n_blocks"], kwargs["n_qubits"])
        
        # for param in model.parameters():
        #     shp = param.data.shape
        #     param.data.flatten()[-1] = set_weights[fold]
        #     param.data.view(shp)
            
        if "lr" in kwargs.keys():
            optimizer = optim(model.parameters(), lr=kwargs["lr"])
        else:
            optimizer = optim(model.parameters(), lr=0.05)
                    
        X_tr, Y_tr = dataclass[train_idx]
        X_te, Y_te = dataclass[test_idx]
        
        dataclass.fit(X_tr.copy())
        X_tr = dataclass.transform(X_tr)
        X_te = dataclass.transform(X_te)
                
        train_set = FoldFactory(X_tr, Y_tr)
        test_set = FoldFactory(X_te, Y_te)
        #Load data
        train_data = DataLoader(train_set,
                                batch_size=batch_size, shuffle=True)
        val_batch_size = min(len(test_set), val_batch_size)
        test_data = DataLoader(test_set,
                                batch_size=val_batch_size, shuffle=True)

        
        
        losses = []
        val_losses = []
        accs = []
        training_acc = []
        gradient_1, gradient_2 = [], []
        outputs = []
        with torch.autograd.set_detect_anomaly(True):
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
                    
                    output = model(x)
                    pred = output.reshape(*y.shape)
                    train_acc = (torch.round(pred)==y).sum().item()/batch_size
                    loss = criterion(pred, y)
                    #Backpropagation
                    if loss.requires_grad:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                        
                    return loss
        
                optimizer.step(loss_closure)
    
                gradient_1.append(model.var[0][0].weights.grad.flatten()[-1].item())
                gradient_2.append(model.var[-1][0].weights.grad.flatten()[-1].item())
                training_acc.append(train_acc)
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
                        _ = model(x_val[idx_l], verbose=True)
                        __ = model(x_val[idx_u], verbose=True)
                        print(pred[idx_l])
                        print(pred[idx_u]%1)
                    
                    #print(epoch, val_loss.item())
                    val_losses.append(val_loss.item())
           
        n_final_t_samples = min(len(train_set), 500)
        final_t_data = DataLoader(train_set, batch_size=n_final_t_samples)                
        x, y = next(iter(final_t_data))
        output = model(x)
        pred = output.reshape(*y.shape)
        y = y.float()
        loss = criterion(pred, y)
        losses.append(loss.item())
        training_acc.append((torch.round(pred)==y).sum().item()/n_final_t_samples)

        n_final_samples = min(len(test_set), 500)
        final_data = DataLoader(test_set, batch_size=n_final_samples)                
        x_val, y_val = next(iter(final_data))
        output = model(x_val)
        pred = output.reshape(*y_val.shape)
        y_val = y_val.float()
        
        val_loss = criterion(pred, y_val)
        val_losses.append(val_loss.item())
        
        accs.append((torch.round(pred)==y_val).sum().item()/n_final_samples)
        results.append({"args":args, "training_loss":losses, "training_acc":training_acc,
                        "val_loss":val_losses, "val_acc":accs, "model":model,
                        "final_preds":{"x_val":x_val, "pred":pred, "y_val":y_val},
                        "gradient1": gradient_1, "gradient2": gradient_2})
        
    return results

if __name__ == "__main__":
    batch_size = 16
    n_blocks = 2
    n_qubits = 5
    n_layers = 2
    epochs = 150
    kfolds = 6
    lr = 0.05
    dataset = "ion"
    reps=1
    results = []
    start = time.time()
    for rep in range(reps):
        print()
        print(rep)
        
        #seed = int(time.time())%100000
        seed = 4321 + rep
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = PQC_4B
        #model = LinearNetwork(10, rep)
        optim = torch.optim.Adam
        R = train(model, optim, dataset, batch_size, epochs, batch_size*2, kfolds, seed,
                  n_blocks=n_blocks, n_qubits=n_qubits, n_layers=n_layers)
        results.append(R)
        # for j in range(kfolds):
        #     plt.plot(R[j]["val_acc"])
        # plt.ylabel("Validation Accuracy")
        # plt.show()
        # for j in range(kfolds):
        #     plt.plot(R[j]["training_acc"])
        # plt.ylabel("Training Accuracy")
        # plt.show()
        # for j in range(kfolds):
        #     plt.plot(R[0]["training_loss"])
        # plt.ylabel("Training Loss")
        # #plt.xlabel("Epoch")
        # plt.show()
        #print([x["val_acc"][-1] for x in R])
        print(np.mean([x["val_acc"][-1] for x in R]))
        print(np.mean([x["training_acc"][-1] for x in R]))
    end = time.time()
    print(end-start)

