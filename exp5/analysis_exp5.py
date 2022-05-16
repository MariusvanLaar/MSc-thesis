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
models=["PQC-4A", "PQC-4B", "PQC-4C"]
dataset = "ion"


# fig1, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12,4))
# for j, b in enumerate(range(2,7,2)):
#     print(b)
#     for i, MOD in enumerate(models):  
#         #print(MOD)
#         layrs_found = []
#         mean_v_acc = []
#         mean_g_err = []
#         fig2, ax2 = plt.subplots()
#         ax2.set_title(f"{b} blocks, model {MOD}")
#         for layer in range(1,8):
#             fname = f"../runs/Exp5*{dataset}*{MOD}-{layer}-{b}*.pkl"
#             files = glob.glob(fname)
#             v_accs = np.zeros((len(files), 16))
#             t_accs = np.zeros((len(files), 151))
#             g_err = np.zeros((len(files)))
#             if len(files) >= 1:
#                 for i, f in enumerate(files):
#                     pickle_open = open(f, 'rb')
#                     run_dict = pickle.load(pickle_open)
#                     v_accs[i] = run_dict["validation_accuracy"]
#                     t_accs[i] = run_dict["training_acc"]
#                     g_err[i] = run_dict["training_acc"][-1] - run_dict["validation_accuracy"][-1]
                                    
#                 ax2.plot(np.arange(20,151), np.mean(t_accs, axis=0)[20:], alpha=0.8, label=f"Training - {layer}")
#                 ax2.plot(10*np.arange(2,16), np.mean(v_accs, axis=0)[2:], alpha=0.8, label=f"Validation - {layer}")
#             mean_v_acc.append(np.mean(v_accs))
#             mean_g_err.append(np.mean(g_err))
#             layrs_found.append(layer)
#         ax2.legend(loc=(1,0))
#         fig2.show()
#         # plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
#         ax[j].plot(layrs_found, mean_v_acc, "x-", label=MOD)
#         ax[j].set_title(f"{b} blocks")
#         #ax[j].plot(layrs_found, mean_g_err, "x-", label=MOD)
# fig1.supxlabel("# Layers")
# ax[0].set_ylabel("Mean validation accuracy")
# #ax[0].set_ylabel("Mean generalization error")
# plt.tight_layout()
# ax[0].legend()

# plt.show()

### Convoluted plot of model performance 

# linetypes = ["-", "--", "-.-"]
# colours = ["b", "r", "g"]
# markers = ["x", "o", "^"]
# plt.figure(figsize=(4,10))
# for i, MOD in enumerate(models):
#     for j, b in enumerate(range(2,7,2)):

#         layrs_found = []
#         mean_v_acc = []
#         mean_t_acc = []
#         for layer in range(2,8):
#             fname = "../runs/Exp5*"+dataset+"*"+str(MOD)+"-"+str(layer)+"-"+str(b)+"*.pkl"
#             files = glob.glob(fname)
#             if len(files) >= 1:
#                 v_accs = np.zeros((len(files)))
#                 t_accs = np.zeros((len(files)))
#                 for k, f in enumerate(files):
#                     pickle_open = open(f, 'rb')
#                     run_dict = pickle.load(pickle_open)
#                     v_accs[k] = run_dict["validation_accuracy"][-1]
#                     t_accs[k] = run_dict["training_acc"][-1]
                                    
#                 mean_v_acc.append(np.mean(v_accs))
#                 mean_t_acc.append(np.mean(t_accs))
#                 layrs_found.append(layer)

#         plt.plot(layrs_found, mean_v_acc, colours[j]+markers[i]+linetypes[0])
#         plt.plot(layrs_found, mean_t_acc, colours[j]+markers[i]+linetypes[1])
#         if i == 0:
#             plt.plot([], [], colours[j]+"-", label=b) 
#     plt.plot([], [], "k"+markers[i], label=MOD)
#         #mean_v_acc = np.array(mean_v_acc)
#         #mean_t_acc = np.array(mean_t_acc)
#         #layrs_found = np.array(layrs_found)

# plt.plot([], [], "k"+linetypes[0], label="Validation set")
# plt.plot([], [], "k"+linetypes[1], label="Training set")
# plt.legend()
# plt.xlabel("Number of layers")
# plt.ylabel("Mean accuracy")
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



### Test the effect of data reuploading
## with ion data
# Models = ["PQC-4A","PQC-4C"]
# Layers = [*range(1,8)]
# Blocks = [2]
# Exps = [5]
# Qubits = [5]

# v_acc, t_acc = [], []
# S, M, L, Q, E = [], [], [], [], []

# for m, l, b, e, q in itertools.product(Models, Layers, Blocks, Exps, Qubits):
#     fname = f"../runs/Exp{e}*ion*-{m}-{l}-{b}-{q}*.pkl"
#     files = glob.glob(fname)
#     for f in files:
#         with open(f, "rb") as pickle_open:
#             run_dict = pickle.load(pickle_open)
#             M.append(m)
#             L.append(l)
#             Q.append(b*q)
#             E.append(e)
#             S.append(f.split("-")[1])
#             v_acc.append(run_dict["validation_accuracy"])
#             t_acc.append(run_dict["training_acc"])


# df = pd.DataFrame(list(zip(S, M, L, Q, E, v_acc, t_acc)), columns=["Seed", "model", "layers", "qubits", "exp",
#                                                                           "v_acc", "t_acc"])
# df.to_csv("reuploading_data.csv")
# df_copy = df
# fig1, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,4))
# for i, MOD in enumerate(df["model"]): 
#     df_ = df[df["model"]==MOD]
#     for l in np.unique(df_["layers"]):
#         df__ = df_[df_["layers"]==l]
#         mean_v_acc.append(np.mean(g_err))
#         mean_g_err.append(np.mean(g_err))
#         layrs_found.append(layer)
            
#     # plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
#     ax[j].plot(layrs_found, mean_v_acc, "x-", label=MOD)
#     ax[j].set_title(f"{b} blocks")
#     #ax[j].plot(layrs_found, mean_g_err, "x-", label=MOD)
# fig1.supxlabel("# Layers")
# ax[0].set_ylabel("Mean validation accuracy")
# #ax[0].set_ylabel("Mean generalization error")
# plt.tight_layout()
# ax[0].legend()
# plt.show()
            

### Compare old models without free Rz rotations to new ones with
## PQC-4D and -4E are same as -4B and -4C respectively but have all-qubit observables
opts=["adam"]
lrs=[0.05]
models=["PQC-4B", "PQC-4D", "PQC-4C", "PQC-4E"]
datasets = ["ion", "spectf", "ion"]
exps = ["", "1"]

for d in datasets:
    for layer in range(2,5):  
        #print(MOD)
        layrs_found = []
        mean_v_acc = []
        mean_g_err = []
        for i, MOD in enumerate(models):
            for exp in exps:
                fname = f"../runs/Exp{exp}5*{d}*{MOD}-{layer}-2-5*.pkl"
                files = glob.glob(fname)
                v_accs = np.zeros((len(files), 1))
                t_accs = np.zeros((len(files), 1))
                g_err = np.zeros((len(files)))
                if len(files) >= 1:
                    for i, f in enumerate(files):
                        pickle_open = open(f, 'rb')
                        run_dict = pickle.load(pickle_open)
                        v_accs[i] = run_dict["validation_accuracy"][-1]
                        t_accs[i] = run_dict["training_acc"][-1]
                    print(f"{d}, {MOD}, {layer}, {exp}5, {np.mean(v_accs)}," \
                          f"{np.mean(t_accs)}, {np.mean(t_accs-v_accs)}")
            
