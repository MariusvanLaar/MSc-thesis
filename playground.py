# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

Z = np.array([[1,0],[0,-1]])
X = np.array([[0,1],[1,0]])
I = np.array([[1,0],[0,1]])

plus = np.array([1, -1j])/np.sqrt(2)

psi0 = np.zeros(8)
psi0[0] = 1


U = np.kron(X,X) + 1j*np.kron(Z,Z)

U1 = np.kron(U,I)
psi2 =np.matmul(U1,psi0) 
print(psi2)
U2 = np.kron(I,U)
psi3 = np.matmul(U2, psi2)
print(psi3)

    




