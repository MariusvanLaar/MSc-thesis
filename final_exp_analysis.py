# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 11:46:51 2022

@author: Marius
"""

import glob
import matplotlib.pyplot as plt
import numpy as np
import pickle

datasets = ["wdbc", "ion", "mnist"]
colors = ["b", "g", "r", "k"]


 
command_train = None
j=0
v_x_range = np.arange(0,151, 10)
t_x_range = np.arange(0,151)
#b, q fixed, l is over range 0 to 7
# for d in datasets:
#     for l in range(8):
#         fig1, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12,4))
#         fig1.suptitle(f"Dataset: {d}, {l} layers")
#         ax[0].set_ylabel("Accuracy")
#         fig1.supxlabel("Epoch")
#         v_acc_mean, v_acc_std = [], []
#         t_acc_mean, t_acc_std = [], []
#         fname = f"runs/Exp20*{d}*-{l}-*-*-First*.pkl"
#         files = glob.glob(fname)
#         if len(files) >= 1:
#             v_accs = np.zeros((len(files), 16))
#             t_accs = np.zeros((len(files), 151))
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 v_accs[i] = run_dict["validation_accuracy"]
#                 t_accs[i] = run_dict["training_acc"]
#             #ax[j].set_title(f"{label_ent(m)}")
#             ax[j].errorbar(v_x_range, np.mean(v_accs, axis=0), yerr=np.std(v_accs, axis=0), fmt="k-")
#             #ax[j].errorbar(t_x_range[1::2], np.mean(t_accs, axis=0)[1::2], yerr=np.std(t_accs, axis=0)[1::2], fmt=colors[j]+"--", alpha=0.6)
#             ax[j].plot(t_x_range, np.mean(t_accs, axis=0), "--", alpha=0.8)
#             ax[j].fill_between(t_x_range, np.mean(t_accs, axis=0)-np.std(t_accs, axis=0), np.mean(t_accs, axis=0)+np.std(t_accs, axis=0), color="y", alpha=0.5)
#         plt.show()
        
# datasets = ["wdbc", "ion"]
# for ind, d in enumerate(datasets):
#     mean_v_acc, std_v_acc = [], []
#     mean_g_err, std_g_err = [], []
#     best_v_acc = []
#     for l in range(8):
#         fname = f"runs/Exp20*{d}*-{l}-{2*(ind+1)}-5-First*.pkl"
#         files = glob.glob(fname)
#         print(len(files))
#         if len(files) >= 1:
#             v_accs = np.zeros((len(files)))
#             t_accs = np.zeros((len(files)))
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 v_accs[i] = run_dict["validation_accuracy"][-1]
#                 t_accs[i] = run_dict["training_acc"][-1]
                
#         mean_v_acc.append(np.mean(v_accs))
#         std_v_acc.append(np.std(v_accs))
#         mean_g_err.append(np.mean(t_accs-v_accs))
#         std_g_err.append(np.std(t_accs-v_accs))
#         best_v_acc.append(np.max(v_accs))
        
#     fig = plt.figure(dpi=300)
#     #Potentially change y tick labels?
#     ax = fig.add_subplot(111)
#     ax2 = ax.twinx()
#     ax.errorbar([*range(8)],mean_v_acc, yerr=std_v_acc, capsize=4, label="Mean", fmt="k-")
#     ax.plot(best_v_acc, "k--", label="Best", )
#     ax.set_ylabel("Validation accuracy")
#     # plt.setp(ax, yticks=[i/2 for i in range(-12,-5)], 
#     #           yticklabels=[r"$10^{-6}$", r"$10^{-5.5}$", r"$10^{-5}$", r"$10^{-4.5}$",
#     #                       r"$10^{-4}$", r"$10^{-3.5}$", r"$10^{-3}$", ])
#     ax.set_xlabel("Number of cut gates")
#     #ax.set_yscale("log")
#     ax2.errorbar([*range(8)],mean_g_err, yerr=std_g_err, fmt="r",
#                   label="Generalization error", capsize=4, alpha=0.8)
#     ax2.tick_params(axis="y", labelcolor="r")
#     if d == "ion":
#         ax2.set_ylim(-0.15,0.3)
#         ax.legend(loc="upper center")
#         plt.grid(axis="y")

#     elif d == "wdbc":
#         ax2.set_ylim(-0.1,0.16)
#         ax.legend(loc=(0.4, 0.75))
#     ax2.set_ylabel("Generalization error")
    
#     plt.show()
    

datasets = ["ising",]
for d in datasets:
    mean_v_l, std_v_l = [], []
    mean_g_l, std_g_l = [], []
    best_v_l = []
    for l in range(8):
        fname = f"runs/Exp20*{d}*-{l}-2-5-0,0*.pkl"
        files = glob.glob(fname)
        print(len(files))
        if len(files) >= 1:
            v_accs = np.zeros((len(files)))
            t_accs = np.zeros((len(files)))
            for i, f in enumerate(files):
                pickle_open = open(f, 'rb')
                run_dict = pickle.load(pickle_open)
                v_accs[i] = np.log10(run_dict["validation_loss"][-1])
                t_accs[i] = np.log10(run_dict["training_loss"][-1])
                
        mean_v_l.append(np.mean(v_accs))
        std_v_l.append(np.std(v_accs))
        mean_g_l.append(np.mean(v_accs-t_accs))
        std_g_l.append(np.std(v_accs-t_accs))
        best_v_l.append(np.min(v_accs))
        
    print(mean_g_l)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.errorbar([*range(8)],mean_v_l, yerr=std_v_l, capsize=4, label="Mean", fmt="k-")
    ax.plot(best_v_l, "k--", label="Best", )
    ax.set_ylabel("Validation loss")
    plt.setp(ax, yticks=[i/2 for i in range(-12,-5)], 
              yticklabels=[r"$10^{-6}$", r"$10^{-5.5}$", r"$10^{-5}$", r"$10^{-4.5}$",
                          r"$10^{-4}$", r"$10^{-3.5}$", r"$10^{-3}$", ])
    ax.set_xlabel("Number of cut gates")
    #ax.set_yscale("log")
    ax2.errorbar([*range(8)],mean_g_l, yerr=std_g_l, fmt="r",
                  label="Generalization error", capsize=4, alpha=0.8)
    ax2.tick_params(axis="y", labelcolor="r")
    ax2.set_ylim(-0.6,0.7)
    ax2.set_ylabel("Generalization loss")
    ax.legend()
    plt.show()
    
# datasets = ["synth-4F"]
# for d in datasets:
#     mean_v_l, std_v_l = [], []
#     mean_g_l, std_g_l = [], []
#     best_v_l = []
#     for l in range(8):
#         fname = f"runs/Exp20*{d}*-{l}-2-5-First*.pkl"
#         files = glob.glob(fname)
#         print(len(files))
#         if len(files) >= 1:
#             v_accs = np.zeros((len(files)))
#             t_accs = np.zeros((len(files)))
#             for i, f in enumerate(files):
#                 pickle_open = open(f, 'rb')
#                 run_dict = pickle.load(pickle_open)
#                 v_accs[i] = run_dict["validation_loss"][-1]
#                 t_accs[i] = run_dict["training_loss"][-1]
#                 print
                
#         mean_v_l.append(np.mean(v_accs))
#         std_v_l.append(np.std(v_accs))
#         mean_g_l.append(np.mean(v_accs-t_accs))
#         std_g_l.append(np.std(v_accs-t_accs))
#         best_v_l.append(np.min(v_accs))
        
#     fig = plt.figure(dpi=300)
#     #Potentially change y tick labels?
#     ax = fig.add_subplot(111)
#     ax2 = ax.twinx()
#     ax.errorbar([*range(8)],mean_v_l, yerr=std_v_l, capsize=4, label="Mean", fmt="k-")
#     ax.plot(best_v_l, "k--", label="Best", )
#     ax.set_ylabel("Validation loss")
#     plt.setp(ax, yticks=[i/10000 for i in range(10,19,2)], 
#               yticklabels=[r"$1.0\times10^{-3}$", r"$1.2\times10^{-3}$", r"$1.4\times10^{-3}$", 
#                             r"$1.6\times10^{-3}$", r"$1.8\times10^{-3}$"])
#     plt.setp(ax2, yticks=[i/10000 for i in range(-20,6,5)], 
#               yticklabels=[r"$-2.0\times10^{-3}$", r"$-1.5\times10^{-3}$", r"$-1.0\times10^{-3}$", 
#                             r"$-0.5\times10^{-3}$", r"$0.0$", r"$0.5\times10^{-3}$"])
#     ax.set_xlabel("Number of cut gates")
#     #ax.set_yscale("log")
#     ax2.errorbar([*range(8)],mean_g_l, yerr=std_g_l, fmt="r",
#                   label="Generalization error", capsize=4, alpha=0.8)
#     ax2.tick_params(axis="y", labelcolor="r")
#     ax2.set_ylim(-7e-4,9e-4)
#     ax2.set_ylabel("Generalization loss")
#     ax.legend(loc=(0.8, 0.2))
#     plt.grid(axis="y")
#     plt.show()



### placement distribution

# cb, ci, tb, ti = [], [], [], []
# for l in range(8):
#     c = 0
#     fname = f"runs/Exp20*synth-4f*-{l}-2-5-First*.pkl"
#     files = glob.glob(fname)
#     for f in files:
#         pickle_open = open(f, 'rb')
#         run_dict = pickle.load(pickle_open)
#         if run_dict["validation_loss"][-1] <= 1.2e-3:
#             c += 1
#             for L in run_dict["CNOT_placement"]: 
#                 for coord in L.coordinates:
#                     cb.append(coord[0][0])
#                     ci.append(coord[0][1])
#                     tb.append(coord[1][0])
#                     ti.append(coord[1][1])
                    
#     print(c)
    
# plt.hist2d(cb, tb, bins=4)
# plt.colorbar()
# plt.show()
# # plt.hist2d(ci, ti, bins=5)
# # plt.colorbar()
# # plt.show()

# cb, ci, tb, ti = [], [], [], []
# for l in range(8):
#     c = 0
#     fname = f"runs/Exp20*ion*-{l}-4-5-First*.pkl"
#     files = glob.glob(fname)
#     for f in files:
#         pickle_open = open(f, 'rb')
#         run_dict = pickle.load(pickle_open)
#         if run_dict["validation_accuracy"][-1] >= 0.92:
#             c += 1
#             for L in run_dict["CNOT_placement"]: 
#                 for coord in L.coordinates:
#                     cb.append(coord[0][0])
#                     ci.append(coord[0][1])
#                     tb.append(coord[1][0])
#                     ti.append(coord[1][1])
#     print(c)
# plt.hist2d(cb, tb, bins=4)
# plt.colorbar()
# plt.show()
# plt.hist2d(ci, ti, bins=5)
# plt.colorbar()
# plt.show()

# cb, ci, tb, ti = [], [], [], []
# for l in range(8):
#     c = 0
#     fname = f"runs/Exp20*ion*-{l}-4-5-First*.pkl"
#     files = glob.glob(fname)
#     for f in files:
#         pickle_open = open(f, 'rb')
#         run_dict = pickle.load(pickle_open)
#         if run_dict["validation_accuracy"][-1] <= 0.92 and run_dict["validation_accuracy"][-1] >= 0.72:
#             c += 1
#             for L in run_dict["CNOT_placement"]: 
#                 for coord in L.coordinates:
#                     cb.append(coord[0][0])
#                     ci.append(coord[0][1])
#                     tb.append(coord[1][0])
#                     ti.append(coord[1][1])
#     print(c)
# plt.hist2d(cb, tb, bins=4)
# plt.colorbar()
# plt.show()
# plt.hist2d(ci, ti, bins=5)
# plt.colorbar()
# plt.show()

        
        
        
