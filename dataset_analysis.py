# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 11:47:26 2022

@author: Marius
"""

import numpy as np 
import torch
import os
import pickle
import matplotlib.pyplot as plt
from main import train
from models import *
import pandas as pd
import datasets

def label(X):
    with torch.no_grad():
        return torch.where(X>0.5, 1, 0)

dataset = "ion"
dataclass = datasets.all_datasets[dataset]()
n_features = dataclass.num_features()
if dataset == "wdbc":
    n_features = 10
data = dataclass.data

pickle_off = open(dataset+"_single_qubit_classifier.pkl", 'rb')
results = pickle.load(pickle_off)
if type(results) == dict:
    results = results["results"]


indexs, means, stds = [], [], []
tmeans = []
for i, R in enumerate(results):
    indexs.append(i)
    val_accs = []
    tra_accs = []
    for r in R:
        val_accs.append(r["val_acc"][-1])
        tra_accs.append(r["training_acc"][-1])
    means.append(np.mean(val_accs))
    stds.append(np.std(val_accs))
    tmeans.append(np.mean(tra_accs))
    
df = pd.DataFrame({"Mean_validation_acc": means, "Std":stds, "Mean training acc": tmeans, "Feature index": indexs})
# print(df.sort_values(by=["Mean"], ignore_index=True))


###Analyse overlap in the predictions from single qubit classifier with one feature
overlap_m = np.ones((n_features,n_features))
for i, R1 in enumerate(results[:-1]):
    for j, R2 in enumerate(results[i+1:]):
        overlap = []
        for r1, r2 in zip(R1, R2):
            # plt.scatter(r1["final_preds"]["pred"].detach(), r2["final_preds"]["pred"].detach())
            preds_1 = label(r1["final_preds"]["pred"])
            preds_2 = label(r2["final_preds"]["pred"])
            assert len(preds_1) == len(preds_2), "Different number of predictions"

            overlap.append(torch.sum(preds_1 == preds_2).item() / len(preds_1))
        # plt.show()
        overlap_m[i,j+i+1] = np.mean(overlap)
        
overlap_m = np.triu(overlap_m) + np.tril(overlap_m.T)
np.fill_diagonal(overlap_m, 1)
fig, (ax1, ax2) = plt.subplots(2,1, sharex="all", gridspec_kw={'height_ratios': [1, 2]})

ax1.bar(np.arange(n_features), df["Mean_validation_acc"])
ax1.set_ylim(0.45, 0.8)
ax1.set_ylabel("Mean validation accuracy")
ax2.tick_params(left=False, labelleft=False)
ax2.set_xlabel("Feature index")
iax2 = ax2.imshow(np.sort(overlap_m,axis=0)[:-1], cmap="bwr")
ax2.set_aspect("auto")
#plt.colorbar(iax2, ax=ax2, location="left")
plt.show()
overlap_m = np.triu(overlap_m) + np.tril(overlap_m.T)
df["Overlap_mean"] = np.mean(overlap_m, axis=0)
#print(df.sort_values(by=["Mean_validation_acc"], ignore_index=True))
plt.scatter(df["Mean_validation_acc"], df["Overlap_mean"])
#plt.xlim(0.48, 0.76)
#plt.ylim(0.48, 0.76)
plt.xlabel("Mean validation accuracy after 4500 training samples")
plt.ylabel("Mean overlap in predictions")
plt.show()

#print(np.mean(overlap_m, axis=0))       
plt.hist(np.mean(overlap_m, axis=0), label="Ion")
plt.title("Mean overlap of each feature")
plt.show()
# print("Mean: {:.2f}, std: {:.2f}".format(np.mean(val_accs), np.std(val_accs)))

### Check Spearmans correlation between features and between model output for features.
from scipy.stats import spearmanr
i_counts, j_counts = [], []
spear_map = np.zeros((n_features,n_features))
for i in range(n_features-1):
    for j in range(i+1, n_features):
        #i = 20
        #j = 23
        data_i = data[:,i]
        data_j = data[:,j]
        res_i = results[i]
        res_j = results[j]
        preds_i = torch.cat([res_i[n]["final_preds"]["pred"].detach() for n in range(len(res_i))])
        preds_j = torch.cat([res_j[n]["final_preds"]["pred"].detach() for n in range(len(res_j))])
        if spearmanr(data_i, data_j)[1] < 0.05:
            if abs(spearmanr(data_i, data_j)[0]) > 0.5:
                i_counts.append(i)
                j_counts.append(j)
                spear_map[i,j] = spearmanr(data_i, data_j)[0]
        #         print(i,j, "datavdata", spearmanr(data_i, data_j))
        # if spearmanr(preds_i, preds_j)[1] < 0.05:
        #     if abs(spearmanr(preds_i, preds_j)[0]) > 0.5:
        #         print(i,j, "predvpred", spearmanr(preds_i, preds_j))
        #         i_counts.append(i)
        #         j_counts.append(j)
        # if spearmanr(data_i, preds_i)[1] < 0.05:
        #     if abs(spearmanr(data_i, preds_i)[0]) > 0.5:
        #         print(i,"i", "datavpred", spearmanr(data_i, preds_i))
        #         i_counts.append(i)
        #         j_counts.append(j)
        # print(spearmanr(preds_i, preds_j))
        # print(spearmanr(data_i, preds_i))
        # print(spearmanr(data_j, preds_j))
        
uni, cou = np.unique(i_counts+j_counts, return_counts=True)
print(uni)
print(cou)
spear_map = np.triu(spear_map) + np.tril(spear_map.T)
plt.figure(figsize=(8,8))
plt.imshow(spear_map)
plt.colorbar()
plt.show()

from sklearn.decomposition import PCA

exp_vars = [0.5, 0.75, 0.9, 0.95, 0.99]
for e_v in exp_vars:
    pca = PCA(n_components=e_v)
    pca.fit(data)
    # f = pca.transform(data)
    print(e_v, pca.n_components_)




