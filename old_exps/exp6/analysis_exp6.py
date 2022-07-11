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

# opts=["adam"]
# lrs=[0.05]
# models=["PQC-5A"]
# dataset = "ion"


# fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
# for i, MOD in enumerate(models):  
#     #print(MOD)
#     layrs_found = []
#     mean_v_acc = []
#     mean_g_err = []
#     for layer in range(1,9):
#         fname = f"../runs/Exp6-*-{dataset}-*{MOD}-{layer}-3-5*.pkl"
#         files = glob.glob(fname)
#         print(len(files))
#         v_accs = np.zeros((len(files)))
#         g_err = np.zeros((len(files)))
#         if len(files) >= 1:
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 v_accs[i] = run_dict["validation_accuracy"][-1]
#                 g_err[i] = run_dict["training_acc"][-1] - run_dict["validation_accuracy"][-1]
                                
#         mean_v_acc.append(np.mean(v_accs))
#         mean_g_err.append(np.mean(g_err))
#         layrs_found.append(layer)
            
#     # plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
#     ax[0].plot(layrs_found, mean_v_acc, "x-", label=MOD)
#     ax[1].plot(layrs_found, mean_g_err, "x-", label=MOD)
# fig1.supxlabel("# cut CNOTs")
# ax[0].set_ylabel("Mean validation accuracy")
# ax[1].set_ylabel("Mean generalization error")
# plt.tight_layout()
# ax[0].legend()
# plt.show()


### Plot gradient variance against number of qubits and layers

Datasets = ["wdbc", "ion", "spectf"]
Models = ["PQC-4A","PQC-4B","PQC-4C","PQC-4D","PQC-4E",]
Layers = [*range(1,9)]
Blocks = [*range(2,7)]
Exps = [5,6,7]
Qubits = [3,4,5,6]

vargrad1, vargrad2 = [], []
D, M, L, Q, E = [], [], [], [], []

for d, m, l, b, e, q in itertools.product(Datasets, Models, Layers, Blocks, Exps, Qubits):
    fname = f"../runs/Exp{e}*{d}*-{m}-{l}-{b}-{q}*.pkl"
    files = glob.glob(fname)
    for f in files:
        with open(f, "rb") as pickle_open:
            run_dict = pickle.load(pickle_open)
            D.append(d)
            M.append(m)
            L.append(l)
            Q.append(b*q)
            E.append(e)
            vargrad1.append(np.std(run_dict["gradient1"])**2)
            vargrad2.append(np.std(run_dict["gradient2"])**2)


df = pd.DataFrame(list(zip(D, M, L, Q, E, vargrad1, vargrad2)), columns=["dataset", "model", "layers", "qubits", "exp",
                                                                          "grad1", "grad2"])
#df.to_csv("gradient_data.csv")
#df = pd.read_csv("gradient_data.csv")
vargrad1 = df["grad1"]
vargrad2 = df["grad2"]
qubits = df["qubits"]
M = df["model"]
L = df["layers"]
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
for model in np.unique(M):
    vargrad1_ = vargrad1[M==model]
    vargrad2_ = vargrad2[M==model]
    qubits_ = qubits[M==model]
    means1, means2 = [], []
    qf = []
    for q in np.unique(qubits_):
        vargrad1__ = vargrad1_[qubits_==q]
        vargrad2__ = vargrad2_[qubits_==q]
        if len(vargrad1_) > 0:
            means1.append(np.mean(vargrad1__))
            qf.append(q)
            means2.append(np.mean(vargrad2__))
    if len(qf) > 1:
        ax[0].semilogy(qf, means1, label=model)
        ax[1].semilogy(qf, means2, label=model)
ax[0].set_ylabel("Gradient variance")
fig1.supxlabel("Number of qubits")
ax[1].legend()
plt.show()

# fig1, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(10,4))
# for model in np.unique(M):
#     vargrad1_ = vargrad1[M==model]
#     vargrad2_ = vargrad2[M==model]
#     layers_ = L[M==model]
#     means1, means2 = [], []
#     lf = []
#     for l in np.unique(layers_):
#         vargrad1__ = vargrad1_[layers_==l]
#         vargrad2__ = vargrad2_[layers_==l]
#         if len(vargrad1_) > 0:
#             means1.append(np.mean(vargrad1__))
#             lf.append(l)
#             means2.append(np.mean(vargrad2__))
#     if len(lf) > 1:
#         ax[0].semilogy(lf, means1, label=model)
#         ax[1].semilogy(lf, means2, label=model)
# ax[0].set_ylabel("Gradient variance")
# ax[0].set_title("Gate 1")
# ax[1].set_title("Gate 2")
# fig1.supxlabel("Number of layers")
# ax[1].legend()
# plt.show()


### Plot gradient variance against number of cut CNOTs

Datasets = ["ion"]
Models = ["PQC-5A"]
Layers = [*range(1,9)]
Blocks = [3]
Exps = [6]
Qubits = [5]

vargrad1, vargrad2 = [], []
D, M, L, Q, E = [], [], [], [], []

for d, m, l, b, e, q in itertools.product(Datasets, Models, Layers, Blocks, Exps, Qubits):
    fname = f"../runs/Exp{e}*{d}*-{m}-{l}-{b}-{q}*.pkl"
    files = glob.glob(fname)
    for f in files:
        with open(f, "rb") as pickle_open:
            run_dict = pickle.load(pickle_open)
            D.append(d)
            M.append(m)
            L.append(l)
            Q.append(b*q)
            E.append(e)
            vargrad1.append(np.std(run_dict["gradient1"])**2)
            vargrad2.append(np.std(run_dict["gradient2"])**2)


df = pd.DataFrame(list(zip(D, M, L, Q, E, vargrad1, vargrad2)), columns=["dataset", "model", "layers", "qubits", "exp",
                                                                          "grad1", "grad2"])
#df.to_csv("gradient_data.csv")
#df = pd.read_csv("gradient_data.csv")
vargrad1 = df["grad1"]
vargrad2 = df["grad2"]
qubits = df["qubits"]
M = df["model"]
L = df["layers"]
fig1, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
for model in np.unique(M):
    vargrad1_ = vargrad1[M==model]
    vargrad2_ = vargrad2[M==model]
    qubits_ = qubits[M==model]
    means1, means2 = [], []
    qf = []
    for q in np.unique(qubits_):
        vargrad1__ = vargrad1_[qubits_==q]
        vargrad2__ = vargrad2_[qubits_==q]
        if len(vargrad1_) > 0:
            means1.append(np.mean(vargrad1__))
            qf.append(q)
            means2.append(np.mean(vargrad2__))
    if len(qf) > 1:
        ax[0].semilogy(qf, means1, label=model)
        ax[1].semilogy(qf, means2, label=model)
ax[0].set_ylabel("Gradient variance")
fig1.supxlabel("Number of qubits")
ax[1].legend()
plt.show()



                                   
            
          

            
            
            
            
