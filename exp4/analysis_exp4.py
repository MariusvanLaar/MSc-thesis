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



    
command_train = None

opts=["adam"]
lrs=[0.05]
models=["PQC-4A", "PQC-4B", "PQC-4C"]

fig1, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12,3))
for i, MOD in enumerate(models):  
    #print(MOD)
    ax1 = ax[i]
    layrs_found = []
    mean_v_acc = []
    mean_t_acc = []
    for layer in range(3,9):
        fname = "../runs/Exp4*"+"synth-4a-10-*"+str(MOD)+"-"+str(layer)+"*.pkl"
        # print(fname)
        files = glob.glob(fname)
        # print(MOD, layer, len(files))
        if len(files) >= 3:
            v_accs = np.zeros((len(files),31))
            t_accs = np.zeros((len(files)))
            for i, f in enumerate(files):
                pickle_open = open(f, 'rb')
                run_dict = pickle.load(pickle_open)
                v_accs[i] = run_dict["validation_accuracy"]
                t_accs[i] = run_dict["training_acc"][-1]
                                    
            #print(LR, np.min(accs[:,-1]))
            mean_v_acc.append(np.mean(v_accs[:,-1]))
            mean_t_acc.append(np.mean(t_accs))
            layrs_found.append(layer)
            
    # plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
    ax1.plot(layrs_found, mean_v_acc, "x-", label="Validation")
    ax1.plot(layrs_found, mean_t_acc, "x-", label="Training")
    ax1.set_title(MOD)
fig1.supxlabel("# Layers")
ax[0].set_ylabel("Mean accuracy after 300 epochs")
plt.tight_layout()
plt.legend()
plt.show()

  
    #print(MOD)
# for layer in range(3,9):
#     for i, MOD in enumerate(models):
#         mean_v_acc = []
#         layrs_found = []
    
#         fname = "../runs/Exp4*"+"synth-4a-10-*"+str(MOD)+"-"+str(layer)+"*.pkl"
#         # print(fname)
#         files = glob.glob(fname)
#         pickle_open = open(files[0], 'rb')
#         run_dict = pickle.load(pickle_open)
#         plt.plot(run_dict["training_acc"], label=MOD)
#     plt.legend()
#     plt.show()

# opts=["adam"]

# for LR in lrs:
#     for MOD in models:
#         for OPT in opts:
#             fname = "runs/Exp1*"+OPT+"-"+str(LR)+"-"+MOD+"*"
#             files = glob.glob(fname)
#             if len(files) != 5:
#                 print(OPT, MOD, LR, len(files))
#             else:
#                 accs = np.zeros((5,25))
#                 for i, f in enumerate(files):
#                     pickle_open = open(f, 'rb')
#                     run_dict = pickle.load(pickle_open)
#                     plt.plot(run_dict["training_loss"])
                                        
#             plt.title(str(LR)+MOD)
#             plt.show()

# means, maxes, stds = [], [], []
# diffs, sq_diffs = [], []
# for n_lay in range(1,7):
#     print()
#     print(n_lay)
#     for LR in lrs:
#         fname = "../runs/Exp4*"+str(LR)+"-"+"PQC-4A"+"-"+str(n_lay)+"*"
#         files = glob.glob(fname)
        
#         pickle_open = open(files[0], 'rb')
#         args = pickle.load(pickle_open)["args"]
#         kfolds=args.kfolds
#         reps_found = len(files)
#         v_acc = []
#         t_acc = []
        
        
#         # if len(files) != kfolds:
#             # print(n_lay, LR, len(files))
#         Lv = np.zeros((reps_found,args.epochs//10 + 1))
#         for i, f in enumerate(files):
#             pickle_open = open(f, 'rb')
#             run_dict = pickle.load(pickle_open)
#             Lv[i] = run_dict["validation_loss"]
#             v_acc.append(run_dict["validation_accuracy"][-1])
#             t_acc.append(run_dict["training_acc"][-1])
#             # plt.plot(run_dict["training_loss"])
#             # plt.show()
               
#         # print(np.round(np.mean(v_acc), 4), np.round(np.max(v_acc), 4))
#         # print(files[np.argmax(v_acc)])
#         diffs.append(np.mean(np.array(t_acc)-np.array(v_acc)))
#         sq_diffs.append(np.mean((np.array(t_acc)-np.array(v_acc))**2))
#         means.append(np.mean(v_acc))
#         maxes.append(np.max(v_acc))
#         stds.append(np.std(v_acc))
        #print()
    # print()

# colors = ["b","g","r","c","m","y",]
# for j in range(6):
#     plt.errorbar(means[j], maxes[j], yerr=stds[j], fmt="v", c=colors[j], label=j+1, alpha=0.8)
# plt.legend(loc=(0.25,0.25))
# plt.ylabel("Highest validation accuracy")
# plt.xlabel("Mean validation accuracy")
# plt.show()

# plt.plot(np.arange(1,7), means)
# plt.show()
# plt.plot(np.arange(1,7), diffs)
# plt.show()
# plt.plot(np.arange(1,7), sq_diffs)
# plt.show()                                        
            
          

            
            
            
            
