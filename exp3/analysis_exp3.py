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
models=["PQC-4A"]

means, maxes, stds = [], [], []
diffs, sq_diffs = [], []
for n_lay in range(1,7):
    print()
    print(n_lay)
    for LR in lrs:
        fname = "../runs/Exp3*"+str(LR)+"-"+"PQC-4A"+"-"+str(n_lay)+"*"
        files = glob.glob(fname)
        
        pickle_open = open(files[0], 'rb')
        args = pickle.load(pickle_open)["args"]
        kfolds=args.kfolds
        reps_found = len(files)
        v_acc = []
        t_acc = []
        
        
        # if len(files) != kfolds:
            # print(n_lay, LR, len(files))
        Lv = np.zeros((reps_found,args.epochs//10 + 1))
        for i, f in enumerate(files):
            pickle_open = open(f, 'rb')
            run_dict = pickle.load(pickle_open)
            Lv[i] = run_dict["validation_loss"]
            v_acc.append(run_dict["validation_accuracy"][-1])
            t_acc.append(run_dict["training_acc"][-1])
            # plt.plot(run_dict["training_loss"])
            # plt.show()
               
        # print(np.round(np.mean(v_acc), 4), np.round(np.max(v_acc), 4))
        # print(files[np.argmax(v_acc)])
        diffs.append(np.mean(np.array(t_acc)-np.array(v_acc)))
        sq_diffs.append(np.mean((np.array(t_acc)-np.array(v_acc))**2))
        means.append(np.mean(v_acc))
        maxes.append(np.max(v_acc))
        stds.append(np.std(v_acc))
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
plt.plot(np.arange(1,7), diffs)
plt.show()
plt.plot(np.arange(1,7), sq_diffs)
plt.show()


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

                                        
            
          

            
            
            
            
