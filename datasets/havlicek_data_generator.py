# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 09:51:20 2022

@author: Marius
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from numpy import pi
import pandas as pd
import math
import pickle
from layers import *
import time

def krons(psi):
    if psi.shape[0] == 2:
        return torch.kron(psi[0], psi[1])
    elif psi.shape[0] > 2:
        return torch.kron(psi[0], krons(psi[1:]))
    elif psi.shape[0] == 1:
        return psi[0]
    else:
        return print("Invalid input")
    
def SU_generator(n, seed=41):
    """
    Generates an SU matrix
    
    Parameters:
    ------------
    n: scalar
        The number of qubits of the circuit
    seed: scalar
        Optional arguement to set the seed of numpy/the RNG
        
    Returns:
    --------
    A SU(2n) numpy array matrix    
    """
    np.random.seed(seed)
    dim = 2**n
    rand_matrix = np.random.normal(size=(dim,dim)) + (0+1j)*np.random.normal(size=(dim,dim))
    H = np.tril(rand_matrix) + np.tril(rand_matrix, -1).T.conj()
    for i in range(dim):
        H[i,i] = np.real(H[i,i])
        
    l, V = np.linalg.eigh(H)
    V /= np.linalg.det(V)**(1/dim)

    return V
    
    
def gen_datapoint(n_features, V):
    batch_size = 1
    n_blocks = 1
    n_qubits = n_features
    
    X = (torch.rand((batch_size, n_blocks, 1, n_qubits,1))-0.5)*np.pi 
    
    state = torch.zeros((batch_size, n_blocks, 1, 2**n_qubits, 1), dtype=torch.cdouble)
    state[:, :, :, :, 0] = 2**(-n_qubits/2)
    
    Ry = Ry_layer(batch_size, n_blocks, n_qubits, X)
    H_l = Hadamard_layer(batch_size, n_blocks, n_qubits)
    
    state = Ry(state)
    
    # for i in range(n_qubits):
    #     CNOT_l = CNOT_layer(n_blocks, n_qubits, [(i,(i+1)%n_qubits)])
    #     weights = torch.zeros((batch_size, n_blocks, 1, n_qubits, 1))
    #     weights[:, 0, 0, (i+1)%n_qubits, 0] = X[:, 0, 0, i]*X[:, 0, 0, (i+1)%n_qubits]
    #     Rz_d2 = Rz_layer(batch_size, n_blocks, n_qubits, weights)
        
    #     state = CNOT_l(state)
    #     state = Rz_d2(state)
    #     state = CNOT_l(state)
        
    # state = H_l(state)
    # state = Rz(state)
    
    # for i in range(n_qubits):
    #     CNOT_l = CNOT_layer(n_blocks, n_qubits, [(i,(i+1)%n_qubits)])
    #     weights = torch.zeros((batch_size, n_blocks, 1, n_qubits, 1))
    #     weights[:, 0, 0, (i+1)%n_qubits, 0] = X[:, 0, 0, i]*X[:, 0, 0, (i+1)%n_qubits]
    #     Rz_d2 = Rz_layer(batch_size, n_blocks, n_qubits, weights)
        
    #     state = CNOT_l(state)
    #     state = Rz_d2(state)
    #     state = CNOT_l(state)
        
    
    
    #Final state:
    state = torch.matmul(V, state)
    
    state = H_l(state)
    state = Ry(state)
    
    Observ = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
    Observ = Observ.repeat(n_qubits,1,1)
    #Observ[-1, 1] = -1
    #If using not a diagonal observable have to change forward() method below too
    Observable = krons(Observ)
    
    Opsi = Observable*state
    O = torch.matmul(state.transpose(3,4).conj(), Opsi)
    return X, O.real

start = time.time()
n = 10
threshold = 0.05
num_samples = 200 #Must be multiple of 4, is total of train and test sets
pos_data = []
neg_data = []
seed = 56789
V = torch.from_numpy(SU_generator(n, seed=seed))
while len(pos_data) + len(neg_data) < num_samples:
    X, O = gen_datapoint(n, V)
    if O < -threshold:
        if len(neg_data) < num_samples // 2:
            neg_data.append(X.view(-1,))
    elif O > threshold:
        if len(pos_data) < num_samples // 2:
            pos_data.append(X.view(-1,))
            
    if  len(pos_data) + len(neg_data) == num_samples//2:
        print(time.time()-start)
        print("Halfway")
            
training_data = pos_data[:num_samples//4] + neg_data[:num_samples//4]
training_labels = [1 for j in range(int(num_samples/4))] + [0 for i in range(int(num_samples/4))]
test_data = pos_data[num_samples//4:] + neg_data[num_samples//4:]
test_labels = [1 for j in range(int(num_samples/4))] + [0 for i in range(int(num_samples/4))]
        
end = time.time()
print(end-start)



synth_data = {"data": training_data, "labels": training_labels}
pickling_on = open("Havlicek-easy-"+str(n)+"-train.pkl","wb")
pickle.dump(synth_data, pickling_on)
pickling_on.close()

synth_data = {"data": test_data, "labels": test_labels}
pickling_on = open("Havlicek-easy-"+str(n)+"-test.pkl","wb")
pickle.dump(synth_data, pickling_on)
pickling_on.close()