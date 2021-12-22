# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:33:24 2021

@author: Marius
"""

import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


"""
Have to justify parameterization of MGs
"""


i = (0+1j)            

def T_m(R):
    T_m = R.T[::2] + i*R.T[1::2]
    return T_m/2

def convert_proj(projector):
    conv = []
    for j, bit in enumerate(projector):
        if bit == 0:
            conv += [j+1, -j-1]
        elif bit == 1:
            conv += [-j-1, j+1]
        else:
            pass
    return conv

def constr_lookup(Tm):
    lookup = np.zeros((2,2,Tm.shape[0], Tm.shape[0]), dtype=complex)
    lookup[0,0] = np.matmul(Tm, np.matmul(H,Tm.T))
    lookup[1,0] = np.matmul(Tm.conj(), np.matmul(H,Tm.T))
    lookup[0,1] = np.matmul(Tm, np.matmul(H,Tm.T.conj()))
    lookup[1,1] = np.matmul(Tm.conj(), np.matmul(H,Tm.T.conj()))
    return lookup

def construct_O(proj, lookup):
    l_O = len(proj)
    O = np.zeros((l_O, l_O), dtype=complex)
    for k in range(l_O):
        for l in range(k+1, l_O):
        #This indexing below won't work for s>0
            i1 = (1-np.sign(proj[k]))//2
            i2 = (1-np.sign(proj[l]))//2
            O[k,l] = lookup[i1, i2, abs(proj[k])-1, abs(proj[l])-1]
            O[l,k] = -O[k,l]
    return O

def test_prob(h):
    Tm = T_m(expm(-4*h))
    lookup = constr_lookup(Tm)
    projs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    convs = [convert_proj(p) for p in projs]
    Os = [construct_O(conv, lookup) for conv in convs]
    return [np.sqrt(np.real(np.linalg.det(O))) for O in Os]


    
n = 3

H = np.identity(2*n, dtype=complex)
H += i*np.diagflat(np.ones(2*n-1),k=1)
H -= i*np.diagflat(np.ones(2*n-1),k=-1)

ps = []
for it in range(1):
    h = np.random.random((2*n,2*n))
    trues = np.ones((2*n,2*n),dtype=bool)
    mask = np.triu(trues, k=1)
    diag = np.identity(2*n,bool)
    h.T[mask] = -h[mask]
    h[diag] = 0
    pt = test_prob(h)
    ps.append(sum(pt))
    print(pt)
    print(pt[0]+pt[3]+pt[5]+pt[6])
    
    
"""
Things that are uncertain:
    The parameterization/constraints on h
    Check the index calling of the lookup table
    H is as described
"""


