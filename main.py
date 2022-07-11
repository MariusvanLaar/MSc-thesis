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
from optimizers import SPSA, CMA, CWD
import pandas as pd
from sklearn.model_selection import KFold


def train(model, optim, data_filename, batch_size, epochs, val_batch_size, kfolds=10, seed=5, **kwargs):
    args = dict(model=model, dataset=data_filename, batch_size=batch_size, epochs=epochs, val_batch_size=val_batch_size, 
                kfolds=kfolds, kwargs=kwargs)
    
    torch.manual_seed(seed+123456789)
    np.random.seed(seed+123456789)

    #Initialize dataclass
    n_features = kwargs["n_blocks"]*kwargs["n_qubits"]
    dataclass = datasets.all_datasets[data_filename](n_features)
    assert "loss" in dataclass.data_info, "Dataclass has no assigned loss function"
    args = dict(**args, **dataclass.data_info)
    #Set loss function
    if args["loss"] == "BCE":
        criterion = nn.BCELoss()
    elif args["loss"] == "MSE":
        criterion = nn.MSELoss()
    elif args["loss"] == "CE":
        criterion = nn.CrossEntropyLoss()
        
    kf = KFold(kfolds, shuffle=True, random_state=seed)
    results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataclass.data)):
        print(fold)
        torch.manual_seed(seed+fold+123456789)
        np.random.seed(seed+fold+123456789)

        if "index" in kwargs.keys():
            model = args["model"](kwargs["n_blocks"], kwargs["n_qubits"],
                                  n_layers=kwargs["n_layers"], index=kwargs["index"],
                                  **kwargs)
        else:
            # model = args["model"](kwargs["n_blocks"], kwargs["n_qubits"], 
            #                       n_layers=kwargs["n_layers"], **kwargs)
            model = args["model"](**kwargs)
            
        if "obs_multiplier" in args.keys():
            model.Observable *= args["obs_multiplier"]
        
        # for param in model.parameters():
        #     shp = param.data.shape
        #     param.data.flatten()[-1] = set_weights[fold]
        #     param.data.view(shp)

        if "lr" in kwargs:
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
                if args["return_probs"]:
                    output = model.return_probability(output)
                pred = output.reshape(*y.shape)

                train_acc = (torch.round(pred)==y).sum().item()/batch_size
                loss = criterion(pred, y)
                #Backpropagation
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    
                return loss
    
            optimizer.step(loss_closure)

            # gradient_1.append(model.var[0][0].weights.grad.flatten()[-1].item())
            # gradient_2.append(model.var[0][0].weights.flatten()[-1].item())
            training_acc.append(train_acc)
            losses.append(loss.item())
            
            if epoch % 10 == 0:        
                with torch.no_grad():
                    x_val, y_val = next(iter(test_data))
                    output = model(x_val)
                    if args["return_probs"]:
                        output = model.return_probability(output)
                    pred = output.reshape(*y_val.shape)
                    y_val = y_val.float()
                    outputs.append((y_val*pred)[0].item())
                    accs.append((torch.round(pred)==y_val).sum().item()/val_batch_size)
                    val_loss = criterion(pred, y_val)
                    val_losses.append(val_loss.item())
                    
        with torch.no_grad():
            n_final_t_samples = min(len(train_set), 5000)
            final_t_data = DataLoader(train_set, batch_size=n_final_t_samples)                
            x, y = next(iter(final_t_data))
            output = model(x)
            if args["return_probs"]:
                output = model.return_probability(output)
            pred = output.reshape(*y.shape)
            y = y.float()
            loss = criterion(pred, y)
            losses.append(loss.item())
            training_acc.append((torch.round(pred)==y).sum().item()/n_final_t_samples)
            
            final_train = {"x_val": x, "pred": pred, "y_val": y}            
                
            n_final_samples = min(len(test_set), 5000)
            final_data = DataLoader(test_set, batch_size=n_final_samples) 
            x_val, y_val = next(iter(final_data))
            output = model(x_val)
            if args["return_probs"]:
                output = model.return_probability(output)
            pred = output.reshape(*y_val.shape)
            y_val = y_val.float()
            val_loss = criterion(pred, y_val)
            val_losses.append(val_loss.item())
            
            accs.append((torch.round(pred)==y_val).sum().item()/n_final_samples)
            
            final_test = {"x_val": x_val, "pred": pred, "y_val": y_val}
            
            if "X_holdout" in args.keys():
                
                holdout_set = FoldFactory(args["X_holdout"], args["Y_holdout"])
                n_final_samples = min(len(holdout_set), 500)
                final_data = DataLoader(holdout_set, batch_size=n_final_samples)
            
                x_val, y_val = next(iter(final_data))
                output = model(x_val)
                if args["return_probs"]:
                    output = model.return_probability(output)
                pred = output.reshape(*y_val.shape)
                y_val = y_val.float()
                val_loss = criterion(pred, y_val)
                val_losses.append(val_loss.item())
                
                final_holdout = {"x_val": x_val, "pred": pred, "y_val": y_val}
            
            else:
                final_holdout = None
            
        results.append({"args":args, "training_loss":losses, "training_acc":training_acc,
                        "val_loss":val_losses, "val_acc":accs,
                        "final_t_preds":final_train, "final_v_preds": final_test,
                        "final_h_preds":final_holdout,
                        "gradient1": gradient_1, "gradient2": gradient_2})
    return results

if __name__ == "__main__":
    batch_size = 64
    n_blocks = 2
    n_qubits = 1
    n_layers = 10
    epochs = 150
    kfolds = 3
    lrs = [0.1]
    dataset = "circle"
    print(dataset)
    reps=1
    results = []
    start = time.time()
    for lr in lrs:
        print(n_layers)
        
        #seed = int(time.time())%100000
        seed = 1238
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        model = PQC_4Z
        #model = NeuralNetwork
        optim = torch.optim.Adam
        R = train(model, optim, dataset, batch_size, epochs, batch_size, kfolds, seed,
                  n_blocks=n_blocks, n_qubits=n_qubits, n_layers=n_layers,
                  input_dim=10, lr=lr, observable="First"
                  )
        results.append(R)
        for j in range(kfolds):
            plt.plot(np.arange(0,151,10),R[j]["val_acc"])
            plt.plot(np.arange(0,151),R[j]["training_acc"])
            #plt.ylim(0.6,1)
            #plt.ylabel("Validation Accuracy")
        plt.show()
        # for j in range(kfolds):
        #     plt.plot(R[j]["training_acc"])
        # plt.ylabel("Training Accuracy")
        # plt.show()
        # for j in range(kfolds):
        #     plt.plot(R[j]["training_loss"])
        #plt.ylim(0,0.1)
        # plt.show()
        # for j in range(kfolds):
        #     plt.figure(dpi=300)
        #     x = R[j]["final_preds"]["pred"].numpy()
        #     y = R[j]["final_preds"]["y_val"].numpy()
        #     plt.plot(x, y, "o")
            # plt.ylabel("Target label")
            # plt.xlabel("Prediction")
            # plt.show()
        #plt.ylim(0,0.004)
        # plt.ylabel("Training Loss")
        #plt.xlabel("Epoch")
        # plt.show()
        # print([x["training_loss"][-1] for x in R])
        # print(np.mean([x["training_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]))
        # print(np.std([x["training_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]))
        # print(np.mean([x["val_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]))
        # print(np.std([x["val_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]))
        
        # for r in R:
        #     print(torch.std(r["final_t_preds"]["pred"]))
        #     print(torch.std(r["final_t_preds"]["y_val"]))
            
        
        # cheetah = [x["training_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]
        # puma = [x["val_loss"][-1] for x in R if x["training_loss"][-1] < 0.1]
        # print(np.mean([y-x for x,y in zip(cheetah, puma)]))
        # print(np.std([y-x for x,y in zip(cheetah, puma)]))
        
        print(np.mean([x["training_acc"][-1] for x in R]))
        print(np.std([x["training_acc"][-1] for x in R]))
        print(np.mean([x["val_acc"][-1] for x in R]))
        print(np.std([x["val_acc"][-1] for x in R]))
    end = time.time()
    print(end-start)
    
    # plt.figure(dpi=300)
    # for j in range(kfolds):
    #     res = R[j]
    #     X = [x[0] for x in res["final_t_preds"]["x_val"]] + [x[0] for x in res["final_v_preds"]["x_val"]]# + [x[0] for x in res["final_h_preds"]["x_val"]]
    #     preds = list(res["final_t_preds"]["pred"]) + list(res["final_v_preds"]["pred"]) #+ list(res["final_h_preds"]["pred"])
    #     Y = list(res["final_t_preds"]["y_val"]) + list(res["final_v_preds"]["y_val"]) #+ list(res["final_h_preds"]["y_val"])
    #     idx = np.argsort(X)
    #     X = np.array(X)[idx]
    #     preds = np.array(preds)[idx]
    #     plt.plot(X, preds)
    #     Y = np.array(Y)[idx]
    #plt.plot(X, Y, "--")
    #plt.xlabel("Rescaled time", fontsize=14)
    #plt.ylabel(r"$\langle Z_{1} \rangle$", fontsize=14)
    
    #plt.figure(dpi=300)
    for j in range(kfolds):
        res = R[j]
        X0 = [x[0] for x in res["final_t_preds"]["x_val"]] + [x[0] for x in res["final_v_preds"]["x_val"]]# + [x[0] for x in res["final_h_preds"]["x_val"]]
        X1 = [x[1] for x in res["final_t_preds"]["x_val"]] + [x[1] for x in res["final_v_preds"]["x_val"]]# + [x[0] for x in res["final_h_preds"]["x_val"]]
        preds = list(res["final_t_preds"]["pred"]) + list(res["final_v_preds"]["pred"]) #+ list(res["final_h_preds"]["pred"])
        Y = list(res["final_t_preds"]["y_val"]) + list(res["final_v_preds"]["y_val"]) #+ list(res["final_h_preds"]["y_val"])
        X0 = np.array(X0)
        X1 = np.array(X1)
        preds = np.array(preds)
        Y = np.array(Y)
        plt.plot(X0[preds>0.5], X1[preds>0.5], "bo")
        plt.plot(X0[preds<=0.5], X1[preds<=0.5], "rx")
        plt.show()
        
        
        
        