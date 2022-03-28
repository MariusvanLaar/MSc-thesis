# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 09:54:26 2022

@author: Marius
"""

import models 
import torch
import pickle
from datafactory import DataFactory
from torch.utils.data import DataLoader
import glob
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

command_train = None

def calc_loss(model, x, y, criterion):
    state, output = model(x)
    pred = output.view(*y.shape)
    return criterion(pred, y)
    
        
def coord_sweep(f, optimized=True):
    pickle_open = open(f, 'rb')
    run_dict = pickle.load(pickle_open)
    model_state = run_dict["model_state_dict"]
    print(run_dict["validation_accuracy"][-1])
    args = run_dict["args"]
    seed = torch.manual_seed(args.seed)
    
    if args.model[:3] == "PQC":
        model = models.model_set[args.model](
            n_blocks=args.n_blocks,
            n_qubits=args.n_qubits,
            weights_spread=args.initial_weights_spread,
            grant_init=args.initialize_to_identity,
            )
    elif args.model[:3] == "ANN":
        model = models.model_set[args.model](
            input_dim=args.n_blocks*args.n_qubits,
            )
        
    if args.loss == "BCE":
        criterion = nn.BCELoss()
    elif args.loss == "MSE":
        criterion = nn.MSELoss()
    elif args.loss == "CE":
        criterion = nn.CrossEntropyLoss()
        
    if optimized:
        model.load_state_dict(model_state)
    model.eval()
    model.randperm = torch.randperm(args.n_blocks*args.n_qubits, generator=seed)
    
    train_data_ = DataFactory(args.dataset, test_train = "train")
    n_train_samples = min(len(train_data_), 500)   
    train_data = DataLoader(train_data_,
                            batch_size=n_train_samples)
    
    test_data_ = DataFactory(args.dataset, test_train = "test")
    n_test_samples = min(len(test_data_), 500)
    test_data = DataLoader(test_data_, batch_size=n_test_samples)
    
    x_tr, y_tr = next(iter(train_data))
    y_tr = y_tr.float()
    
    x_te, y_te = next(iter(test_data))
    y_te = y_te.float()
    
    thetas = np.linspace(-np.pi, np.pi, 100)
    
    train_loss, test_loss = [], []
    with torch.no_grad():
        
        init_param = list(model.parameters())[2].flatten()[-1].item()
        
        tr_loss = calc_loss(model, x_tr, y_tr, criterion)
        train_loss.append(tr_loss.item())
        
        te_loss = calc_loss(model, x_te, y_te, criterion)
        test_loss.append(te_loss.item())
        
        model_params = list(model.parameters())
        fr2 = model_params[2]
        param_shape = fr2.shape
        
        for T in thetas:
            
            fr2.flatten()[-1] = T
            fr2 = fr2.view(param_shape)
                            
            tr_loss = calc_loss(model, x_tr, y_tr, criterion)
            train_loss.append(tr_loss.item())
            
            te_loss = calc_loss(model, x_te, y_te, criterion)
            test_loss.append(te_loss.item())
            
    plt.plot([init_param]+list(thetas), train_loss, 'bx', label="Training set")
    plt.plot([init_param]+list(thetas), test_loss, 'go', label="Test set")
    if optimized:
        plt.vlines(init_param, 0, 2, label="Optimized weight value")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Weight value")
    plt.title(args.tag+" "+args.model)
    lower_bound = min(min(train_loss), min(test_loss))*0.97
    upper_bound = max(max(train_loss), max(test_loss))*1.02
    plt.ylim(lower_bound, upper_bound)
    plt.show()
        
select_runs = glob.glob('select_runs/*')
for f in select_runs:
    coord_sweep(f, optimized=True)
    coord_sweep(f, optimized=False)
    
    