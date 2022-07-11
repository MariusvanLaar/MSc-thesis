# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:12:52 2022

@author: Marius
"""

import pickle
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation as mad
import pandas as pd
import itertools


    
command_train = None

opts=["adam"]
lrs=[0.05]
models=["PQC-4D", "PQC-4E", "PQC-4C"]
dataset = "wdbc"

V_ACCS = []

# fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
for i, MOD in enumerate(models):  
    print(MOD)
    layrs_found = []
    mean_v_acc = []
    mean_g_err = []
    for layer in range(1,9):
        fname = "../runs/Exp7*"+dataset+"*"+str(MOD)+"-"+str(layer)+"-2"+"*.pkl"
        files = glob.glob(fname)
        v_accs = np.zeros((len(files)))
        g_err = np.zeros((len(files)))
        if len(files) >= 1:
            for i, f in enumerate(files):
                pickle_open = open(f, 'rb')
                run_dict = pickle.load(pickle_open)
                v_accs[i] = min(run_dict["validation_accuracy"][-1], run_dict["training_acc"][-1])
                g_err[i] = run_dict["training_acc"][-1] - run_dict["validation_accuracy"][-1]
                                
        mean_v_acc.append(np.mean(v_accs))
        mean_g_err.append(np.mean(g_err))
        layrs_found.append(layer)
            
    # plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
    V_ACCS.append(mean_v_acc)
#     ax[0].plot(layrs_found, mean_v_acc, "-", label=MOD)
#     ax[1].plot(layrs_found, mean_g_err, "-", label=MOD)
# fig1.supxlabel("# Layers")
# ax[0].set_ylabel("Mean validation accuracy")
# ax[1].set_ylabel("Mean generalization error")
# plt.tight_layout()
# ax[0].legend()
# plt.show()

plt.plot(np.arange(1,9),np.array(V_ACCS[0])-np.array(V_ACCS[-1]), label="Odd")
plt.plot(np.arange(1,9),np.array(V_ACCS[1])-np.array(V_ACCS[-1]), label="Even")
plt.legend()
plt.xlabel("Number of layers")
plt.ylabel("Difference with PQC-4C")
plt.show()

# for i, MOD in enumerate(models):  
#     print(MOD)
#     layrs_found = []
#     mean_v_acc = []
#     for layer in range(1,9):
#         fname = "../runs/Exp7*"+dataset+"*"+str(MOD)+"-"+str(layer)+"-2"+"*.pkl"
#         files = glob.glob(fname)
#         v_accs = np.zeros((len(files)))
#         g_err = np.zeros((len(files)))
#         if len(files) >= 1:
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 v_accs[i] = min(run_dict["validation_accuracy"][-1], run_dict["training_acc"][-1])
                                
#         mean_v_acc.append(np.mean(v_accs))
#         layrs_found.append(layer)
            
#     plt.plot(layrs_found, mean_v_acc, "-", label=MOD)
# plt.xlabel("# Layers")
# plt.ylabel("Mean of max(validation accuracy, training accuracy)")
# plt.legend()
# plt.show()


### Plot gradient variance against number of qubits and layers

# datasets = ["wdbc"]
# models = ["PQC-4E"]
# layers = [*range(1,9)]
# blocks = [2]

# qubits, layers_found = [], []
# vargrad1, vargrad2 = [], []

# for d, m, l, b in itertools.product(datasets, models, layers, blocks):
#     fname = "../runs/Exp7*"+str(d)+"*-"+m+"-"+str(l)+"-"+str(b)+"*.pkl"
#     files = glob.glob(fname)
#     for f in files:
#         pickle_open = open(f, 'rb')
#         run_dict = pickle.load(pickle_open)
#         qubits.append(b*5)
#         layers_found.append(l)
#         vargrad1.append(np.std(run_dict["gradient1"])**2)
#         vargrad2.append(np.std(run_dict["gradient2"])**2)

# vargrad1 = np.array(vargrad1)
# vargrad2 = np.array(vargrad2)
# qubits = np.array(qubits)
# means1, means2 = [], []
# for q in np.unique(qubits):
#     vargrad1_ = vargrad1[qubits==q]
#     vargrad2_ = vargrad2[qubits==q]
    
#     means1.append(np.mean(vargrad1_))
#     means2.append(np.mean(vargrad2_))
    
# plt.semilogy(np.unique(qubits), means1, label="Gradient 1")
# plt.semilogy(np.unique(qubits), means2, label="Gradient 2")
# plt.ylabel("Variance")
# plt.xlabel("Number of qubits")
# plt.legend()
# plt.show()

# layers_found=np.array(layers_found)
# means1, means2 = [], []
# for l in np.unique(layers_found):
#     vargrad1_ = vargrad1[layers_found==l]
#     vargrad2_ = vargrad2[layers_found==l]
    
#     means1.append(np.mean(vargrad1_))
#     means2.append(np.mean(vargrad2_))
    
# plt.semilogy(np.unique(layers_found), means1, label="Gradient 1")
# plt.semilogy(np.unique(layers_found), means2, label="Gradient 2")
# plt.xlabel("Number of layers")
# plt.ylabel("Variance")
# plt.legend()
# plt.show()



### Assess training loss

# datasets = ["ion"]
# models = ["PQC-4C", "PQC-4A"]
# layers = [*range(3,8)]
# blocks = [2,4,6]

# models = ["PQC-4B"]
# for layer in range(3,8):
#     for i, MOD in enumerate(models):
#         for d, b in itertools.product(datasets, blocks):
#             fname = "../runs/Exp5*"+str(d)+"*-"+MOD+"-"+str(layer)+"-"+str(b)+"*.pkl"

#             files = glob.glob(fname)
#             if len(files) >= 1:
#                 pickle_open = open(files[0], 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 plt.plot(run_dict["training_loss"], label=MOD)
#             break
#     plt.legend()
#     plt.show()


# colours = ["b", "r", "g"]
# markers = ["x", "o", "^"]
# plt.figure(figsize=(6,6))
# for i, MOD in enumerate(models):
#     for j, b in enumerate(range(2,7,2)):

#         layrs_found = []
#         mean_g_err = []
#         for layer in range(2,8):
#             fname = "../runs/Exp5*"+dataset+"*"+str(MOD)+"-"+str(layer)+"-"+str(b)+"*.pkl"
#             files = glob.glob(fname)
#             if len(files) >= 1:
#                 g_err = np.zeros((len(files)))
#                 for k, f in enumerate(files):
#                     pickle_open = open(f, 'rb')
#                     run_dict = pickle.load(pickle_open)
#                     g_err[k] = run_dict["training_acc"][-1] - run_dict["validation_accuracy"][-1]
                                    
#                 mean_g_err.append(np.mean(g_err))
#                 layrs_found.append(layer)

#         plt.plot(layrs_found, mean_g_err, colours[j]+markers[i]+"-")
#         if i == 0:
#             plt.plot([], [], colours[j]+"-", label=b) 
#     plt.plot([], [], "k"+markers[i], label=MOD)


# plt.legend()
# plt.ylabel("Mean (training accuracy - validation accuracy)")
# plt.xlabel("Number of layers")
# plt.show()



                                   
            
          

            
            
            
            
