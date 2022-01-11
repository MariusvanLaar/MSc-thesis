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

plus = np.array([1, -1j])/np.sqrt(2)

U = (X+np.sqrt(1j)*Z)/(2*np.sqrt(2))
psi2 =np.matmul(U,plus) 
print(psi2)
print(np.linalg.norm(psi2))

    




