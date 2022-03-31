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
    
command_train = None

opts=["adam"]
lrs=[0.05]
models=["PQC-4A"]#["PQC-3V", "PQC-3W", "PQC-3X", "PQC-3Y", "PQC-3Z", "PQC-4A"]
reps=10

for i, MOD in enumerate(models):
    print()
    print(MOD)
    for OPT in opts:
        std_vloss = []
        for LR in lrs:
            print(LR)
            fname = "../select_runs/Exp2*"+OPT+"-"+str(LR)+"-"+MOD+"*"
            files = glob.glob(fname)
            print(len(files))
            pickle_open = open(files[0], 'rb')
            args = pickle.load(pickle_open)["args"]
            kfolds=args.kfolds
            reps_found = len(files)
            v_acc = []
            
            Lv = np.zeros((reps_found,args.epochs//10 + 1))
            for i, f in enumerate(files):
                pickle_open = open(f, 'rb')
                run_dict = pickle.load(pickle_open)
                Lv[i] = run_dict["validation_loss"]
                v_acc.append(run_dict["validation_accuracy"][-1])
            std_vloss.append(np.std(Lv[:,-1]))
            #plt.show()
            print(np.round(v_acc, decimals=3))
            print(np.round(np.mean(v_acc), 2), np.round(np.max(v_acc), 2))
            print(files[np.argmax(v_acc)])
            print()
        print(np.round(std_vloss, 2))
        print()
        
        
fname = "../select_runs/Exp2*"+"adam"+"-"+"0.05"+"-"+"PQC-4A"+"*"
files = glob.glob(fname)
print(len(files))
pickle_open = open(files[0], 'rb')
run = pickle.load(pickle_open)
        
# lrs=[0.01, 0.05]
        
# for MOD in models:
#     for OPT in opts:
#         for LR in lrs:
#             print(MOD, LR)
#             fname = "../runs/Exp2*"+OPT+"-"+str(LR)+"-"+MOD+"*"
#             files = glob.glob(fname)
            
#             pickle_open = open(files[0], 'rb')
#             args = pickle.load(pickle_open)["args"]
#             kfolds=args.kfolds
#             reps_found = len(files)
#             v_acc = []
            
            
#             if len(files) != kfolds:
#                 print(OPT, MOD, LR, len(files))
#             Lv = np.zeros((reps_found,args.epochs//10 + 1))
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 plt.plot(run_dict["training_loss"])
                
#             plt.show()



# opts=["cma"]

# #fig1, ax = plt.subplots(nrows=1, ncols=6, sharex=True, sharey=True, figsize=(12,3))
# fig1, ax = plt.subplots(nrows=1, ncols=1)
# for i, MOD in enumerate(models):  
#     #print(MOD)
#     #ax1 = ax[i]
#     ax1 = ax
#     for OPT in opts:
#         mean_v_acc = []
#         lrs_found = []
#         for LR in lrs:
        
#             fname = "runs/Exp1*"+OPT+"-"+str(LR)+"-"+MOD+"*"
#             files = glob.glob(fname)
#             # if len(files) != 5:
#             #     print(OPT, MOD, LR, len(files))
#             if len(files) >= 3:
#                 accs = np.zeros((len(files),26))
#                 for i, f in enumerate(files):
#                     pickle_open = open(f, 'rb')
#                     run_dict = pickle.load(pickle_open)
#                     accs[i] = run_dict["validation_accuracy"]
                                        
#                 #print(LR, np.min(accs[:,-1]))
#                 mean_v_acc.append(np.min(accs[:,-1]))
#                 lrs_found.append(LR)
            
#         #plt.errorbar(lrs, mean_v_acc, yerr=np.std(mean_v_acc), label=MOD, alpha=0.75)
#         ax1.plot(lrs_found, mean_v_acc, "x-", label=MOD)
#     ax1.set_ylim(0.6, 1.)
#     ax1.set_xscale("log")
#     ax1.set_title("")
# ax1.legend(loc="lower center")
# fig1.supxlabel("Learning rate")
# #ax[0].set_ylabel("Mean validation accuracy after 250 epochs")
# ax.set_ylabel("Mean validation accuracy after 250 epochs")
# plt.tight_layout()
# plt.show()

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

                                        
            
          

            
            
            
            
