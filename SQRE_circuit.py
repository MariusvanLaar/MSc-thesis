# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 16:53:16 2021

@author: Marius
"""

import numpy as np
import time
import matplotlib.pyplot as plt

class Qubits(object):
    def __init__(self, n):
        self.n = n
        self.decomp_states = [[np.identity(2) for N in range(n)]]
        self.derivatives = []
        
    def R_x(self, Thetas, free=True):
        i = 0+1j
        def rx(angle, comp_der):
            if not comp_der:
                return np.array([[np.cos(angle/2), -i*np.sin(angle/2)],[-i*np.sin(angle/2), np.cos(angle/2)]])
            else:
                return (-i/2)*np.array([[-i*np.sin(angle/2), np.cos(angle/2)],[np.cos(angle/2), -i*np.sin(angle/2)]])
        new_decomp_states = [[np.matmul(rx(theta, comp_der=False), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in self.decomp_states]
        self.derivatives = [[[np.matmul(rx(theta, comp_der=False), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in fparam_string]
                                    for fparam_string in self.derivatives]
        if free:
            self.derivatives.append([[np.matmul(rx(theta, comp_der=True), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in self.decomp_states])
        
        self.decomp_states = new_decomp_states
        
    def R_z(self, Thetas, free=True):
        i = 0+1j
        def rz(angle, comp_der):
            if not comp_der:
                return np.array([[np.exp(-i*angle/2),0],[0,np.exp(i*angle/2)]])
            else:
                return (-i/2)*np.array([[np.exp(-i*angle/2),0],[0,-np.exp(i*angle/2)]])
        new_decomp_states = [[np.matmul(rz(theta, comp_der=False), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in self.decomp_states]
        self.derivatives = [[[np.matmul(rz(theta, comp_der=False), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in fparam_string]
                                    for fparam_string in self.derivatives]
        if free:    
            self.derivatives.append([[np.matmul(rz(theta, comp_der=True), qubit) for (qubit, theta) in zip(qubits, Thetas)] for qubits in self.decomp_states])
        
        self.decomp_states = new_decomp_states
        
    def Entangle(self, indices):
        i = (0+1j)
        X = np.array([[0,1],[1,0]])*(2**-0.25)
        Ze = np.sqrt(i)*np.array([[1,0],[0,-1]])*(2**-0.25)
        for i,j in indices:
            new_decomp_states = []
            for qubit_string in self.decomp_states:
                new_decomp_states.append([np.matmul(X, qubit) if k==i or k==j else qubit for k, qubit in enumerate(qubit_string)])
                new_decomp_states.append([np.matmul(Ze, qubit) if k==i or k==j else qubit for k, qubit in enumerate(qubit_string)])
            self.decomp_states = new_decomp_states
            for a, fparam_string in enumerate(self.derivatives):
                new_derivative_states = []
                for qubit_string in fparam_string:
                    new_derivative_states.append([np.matmul(X, qubit) if k==i or k==j else qubit for k, qubit in enumerate(qubit_string)])
                    new_derivative_states.append([np.matmul(Ze, qubit) if k==i or k==j else qubit for k, qubit in enumerate(qubit_string)])
                self.derivatives[a] = new_derivative_states
            
    def final_state(self):
        def krons(A):
            if len(A)==2:
                return np.kron(A[0][:,0],A[1][:,0])
            elif len(A)==1:
                return A[0][:,0]
            else:
                return np.kron(A[0][:,0], krons(A[1:]))
        f_state = np.sum([krons(qubit_string) for qubit_string in self.decomp_states],axis=0)

        return f_state
    
    def exp_val_Z(self, unitary):
        Z = np.array([1,-1], dtype=complex)
        return np.real((unitary[:,0]*unitary[:,0].conj()*Z).sum())
    
    def Exp_val(self):
        return sum([np.product([self.exp_val_Z(U) for U in decomp]) for decomp in self.decomp_states])        
    
    def DE_Dt(self):
        exp_n = [[[self.exp_val_Z(U) for U in decomp] for decomp in fparam] for fparam in self.derivatives]
        return np.array(exp_n).sum(axis=1).flatten()
   
    
def Circuit(n, x, Thetas, take_derivative=True):
    qubits = Qubits(n)
    qubits.R_x(x[:n], free=False)
    #qubits.R_z(x[n:2*n], free=False)
    #qubits.Entangle([(0,1)])
    for l in range(2):
        qubits.R_x(Thetas[l*n:(l+1)*n], free=take_derivative)
        qubits.R_z(Thetas[(l+1)*n:2*(l+1)*n], free=take_derivative)
    qubits.Entangle([(0,1)])
    for l in range(2,4):
        qubits.R_x(Thetas[l*n:(l+1)*n], free=take_derivative)
        qubits.R_z(Thetas[(l+1)*n:2*(l+1)*n], free=take_derivative)
    
    return qubits

def Cost(E, y):
    return (y-E)*y
    
if __name__ == "__main__":
    costs = []
    correctomundo = []
    mag_de = []
    n=2
    iterations = 10000
    nu = 0.005
    X = [[0,0],[1,1]]
    Y = [1,-1]
    Params = np.random.random((4*n*2))*2*np.pi #Maybe set to normal distribution with mean 0 and small std
    for it in range(iterations):
        correct = 0
        dParams = np.zeros((4*n*2))
        C = 0
        for x, y in zip(X,Y):
            qubits = Circuit(n, x, Params)
            E = qubits.Exp_val()
            if np.sign(E) == y:
                correct += 0.5
            C += Cost(E, y)
            dParams += -y*qubits.DE_Dt()
        dParams /= len(X)
        mag_de.append(sum(np.abs(dParams)))
        Params -= nu*dParams
        costs.append(C)
        correctomundo.append(correct)
        
    plt.plot(costs, 'g-')
    plt.plot(correctomundo, 'bx')
    plt.plot(mag_de, 'k-')

    # times = []
    # for n in N:
    #     start = time.time()
    #     qubits = Qubits(n)
    #     indices_1 = [(i,i+1) for i in range(0,n-1,2)]
    #     #indices_2 = [(i,i+1) for i in range(1,n-1,2)]
    #     for l in range(4):
    #         qubits.R_x(np.random.random((n))*np.pi +2)
    #         qubits.R_z(np.random.random((n))*np.pi +2)
    #     qubits.Entangle(indices_1)
    #     for l in range(4):
    #         qubits.R_x(np.random.random((n))*2*np.pi +2)
    #         qubits.R_z(np.random.random((n))*2*np.pi +2)
    #         #qubits.Entangle(indices_2)
    #     mid = time.time()
    #     #f_state = qubits.final_state()
    #     end = time.time()
    #     times.append(end-start)
    
    
    # times1 = []
    # for n in N:
    #     start = time.time()
    #     qubits = Qubits(n)
    #     indices_1 = [(0,1), (2,3)]#[(i,i+1) for i in range(0,n-1,2)]
    #     indices_2 = []#[(i,i+1) for i in range(1,n-1,2)]
        
    #     #qubits.Entangle(indices_1)
    #     for l in range(4):
    #         qubits.R_x(np.random.random((n))*np.pi +2)
    #         qubits.R_z(np.random.random((n))*np.pi +2)
    #         qubits.R_x(np.random.random((n))*2*np.pi +2)
    #         qubits.R_z(np.random.random((n))*2*np.pi +2)
    #         #qubits.Entangle(indices_2)
    #     mid = time.time()
    #     #f_state = qubits.final_state()
    #     end = time.time()
    #     times1.append(end-start)
        
    # plt.plot(N, times, 'g-')
    # plt.plot(N, times1, 'r-')
    # plt.show()