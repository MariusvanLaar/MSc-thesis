# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:09:16 2021

@author: Marius
"""

import cirq
import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from scipy.linalg import schur, expm

def G(A,B):
    U = np.zeros((4,4), dtype=complex)
    U[::3, ::3] = A
    U[1:3,1:3] = B
    return U

i = (0+1j)
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
Y = np.array([[0,-1],[1,0]])*(0+1j)
I = np.identity(2)

c1 = np.kron(X,I)
c2 = np.kron(Y,I)
c3 = np.kron(Z,X)
c4 = np.kron(Z,Y)
cs = [c1, c2, c3, c4]

def gen_G(x):
    t1 = np.kron(expm(i*x[0]*Z),expm(i*x[1]*Z))
    t2 = expm(i*(x[2]*np.kron(X,X)+x[3]*np.kron(Y,Y)))
    t3 = np.kron(expm(i*x[4]*Z),expm(i*x[5]*Z))
    return np.matmul(t1, np.matmul(t2,t3))

def C(n, p):
    ### p has to be integer in the range [1,2n]
    if p < 1:
        return None
    def krons(N, Pauli):
        if N == 1:
            return Pauli
        elif N == 0:
            return 1
        else:
            return np.kron(Pauli, krons(N-1, Pauli))
    
    if p % 2 == 1:
        Is = krons(n-(p+1)/2, I)
        Zs = krons((p+1)/2-1, Z)
        return np.kron(Zs, np.kron(X,Is))
    if p % 2 == 0:
        Is = krons(n-p/2, I)
        Zs = krons(p/2-1, Z)
        return np.kron(Zs, np.kron(Y,Is))

def trip(a):
    if len(a) == 2:
        return np.matmul(a[0], a[1])
    else:
        return np.matmul(a[0], trip(a[1:]))
    
def krons(a):
    if len(a) == 2:
        return np.kron(a[0],a[1])
    else:
        return np.kron(a[0], krons(a[1:]))
        
seed = 5
rng = np.random.default_rng(seed)
y1 = rng.random(6)
y2 = rng.random(6)

A = np.kron(I,gen_G(y1))
B = np.kron(gen_G(y2),I)

# n = 3
# sup_pos = np.zeros((2**n, 2**n), dtype=complex)
# cs = [C(n,i) for i in range(1,2*n+1)]

# for b in range(1,2):
#     print(np.matmul(A, np.matmul(C(3,b), B)).round(2))
#     print()
for c in cs:
    for k in cs:
        print(np.matmul(c,k))


# for k in [-1,1]:
#     for l in range(0,2*n):
#         ls = [x for x in range(6) if x!=l]
#         css = [cs[i] for i in ls]
        #print(k*cs[l]+trip(css))
#print(sup_pos)

def exp_val(U, f, s):
    return np.dot(s.conj().T, np.dot(U.conj().T,np.dot(f,np.dot(U,s))))



def swap_gadget(n, s):
    return np.log10((4**(2*s)))*np.log10((1728*(s**3))+8*(n**3))

def heisenberg(n,s):
    return np.log10((4**((2*s)-0.5)))*np.log10((n**(2+(5*s))))

N = np.arange(200, 10001, 2, dtype=float)

s = 3.0
csg = [3*swap_gadget(n,s) for n in N]
heis = [heisenberg(n,s) for n in N]

plt.plot(N, csg, 'g-')
plt.plot(N,heis, 'r-')
plt.show()

    

    




