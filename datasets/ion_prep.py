# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 10:15:27 2022

@author: Marius
"""

import pickle
import numpy as np



# filepath = "datasets/ionosphere.data"
# file = open(filepath, "r")

# data = np.zeros((351, 34))
# labels = np.zeros((351))

# for j in range(351):
#     line = file.readline()
#     L  = line.split(",")
#     data[j] = np.array([float(x) for x in L[:-1]])   
#     if str(L[-1]) == "g\n":
#         labels[j] = 1
#     elif str(L[-1]) == "b\n":
#         labels[j] = 0
        
# print(np.unique(labels, return_counts=True))
        
# print(np.std(data, axis=0))
# # print(np.unique(labels))
# data = np.delete(data, 1, 1) #Remove 2nd feature with no variance
# print(np.std(data, axis=0))

# fname = "ion"

# synth_data = {"data": data, "labels": labels}
# pickling_on = open(fname+".pkl","wb")
# pickle.dump(synth_data, pickling_on)
# pickling_on.close()


# labels = np.array(labels, dtype=bool)
# from sklearn.manifold import TSNE

# x_embedded = TSNE(n_components=2).fit_transform(data)
# plt.scatter(x_embedded[:, 0], x_embedded[:, 1], color=['green' if label else 'red' for label in labels])
# plt.show()
