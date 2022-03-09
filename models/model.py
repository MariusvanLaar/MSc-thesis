# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from .layers import Rx_layer, Ry_layer, Rz_layer, Hadamard_layer, CNOT_layer, Entangle_layer, krons
import torch 
import torch.nn as nn
import numpy as np
from functools import reduce
import operator
import math

def Y_LAYER(n_blocks, n_qubits, weights_spread):
    return [
        Ry_layer(n_blocks, n_qubits, weights_spread=weights_spread),
        ]

def ZYZ_LAYER(n_blocks, n_qubits, weights_spread):
    return [
        Rz_layer(n_blocks, n_qubits, weights_spread=weights_spread),
        Ry_layer(n_blocks, n_qubits, weights_spread=weights_spread),
        Rz_layer(n_blocks, n_qubits, weights_spread=weights_spread),
        ]
    
def CNOT(n_blocks, n_qubits, offset, leap):
    return [
        CNOT_layer(n_blocks, n_qubits, [((i+offset)%n_qubits, (i+offset+leap)%n_qubits) for i in range(n_qubits)])
        ]

class BaseModel(nn.Module):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__()   
        
        self.n_blocks = n_blocks
        self.n_qubits = n_qubits
        self.weights_spread = weights_spread
        self.randperm = torch.randperm(n_blocks*n_qubits)
        
        Observ = torch.Tensor([[1,-1]]).cdouble().reshape((1,2,1))
        Observ = Observ.repeat(n_qubits,1,1)
        #Observ[-1, 1] = -1
        #If using not a diagonal observable have to change forward() method below too
        self.Observable = krons(Observ)
        
    def cnot_(self, offset, leap):
        return nn.Sequential(
                CNOT_layer(self.n_blocks, self.n_qubits, 
                           [((i+offset)%self.n_qubits, (i+offset+leap)%self.n_qubits) for i in range(self.n_qubits)])
                )    
    
    def AfRot(self):
        return nn.Sequential(
                Rz_layer(self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Ry_layer(self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                Rz_layer(self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                )  
    
    def YfRot(self):
        return nn.Sequential(
                Ry_layer(self.n_blocks, self.n_qubits, weights_spread=self.weights_spread),
                )
        
    def copy_weights(self, control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
                
    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")
        
    def exp_val(self, state):
        batch_size = state.shape[0]
        O = torch.zeros(batch_size).cdouble()
        a0 = torch.complex(torch.Tensor([1,0]), torch.Tensor([0,-1])) / ((2**0.5)*(-1j)**0.5)
        alpha = torch.Tensor([1]).cdouble()
        for j in range(int(math.log2(state.shape[2]))):
            alpha = torch.kron(alpha, a0)
        for k in range(state.shape[2]):
            a_conj = torch.roll(alpha, k).conj()
            state_conj = torch.roll(state, k, dims=2).transpose(3,4).conj()
            inn_prods = torch.matmul(state_conj, self.Observable*state)
            coefs = (a_conj*alpha).view(1,-1)
            Os = inn_prods.prod(dim=1).view(batch_size, -1) * coefs
            O += Os.sum(dim=1)
        return 0.5*(O.float()+1)
    
class TestModel(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0 = self.AfRot()
        self.fR1 = self.AfRot()
        self.fR2 = self.AfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]]],
                                        self.n_qubits)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        Ry_data = Ry_layer(self.n_blocks, self.n_qubits, weights=x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        #state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)
        state = self.Entangle(state)
                
        return state, self.exp_val(state)

        
class PQC_1A(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0 = self.AfRot()
        self.fR1 = self.AfRot()
        self.fR2 = self.AfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        Ry_data = Ry_layer(self.n_blocks, self.n_qubits, weights=x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)
        state = self.fR0(state)
        state = self.cnot(state)
        state = self.fR1(state)
        state = self.Entangle(state)
        state = self.fR2(state)
                
        return state, self.exp_val(state)
    
class PQC_1Y(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0 = self.YfRot()
        self.fR1 = self.YfRot()
        self.fR2 = self.YfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        Ry_data = Ry_layer(self.n_blocks, self.n_qubits, weights=x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)
        state = self.fR0(state)
        state = self.cnot(state)
        state = self.fR1(state)
        state = self.Entangle(state)
        state = self.fR2(state)
                
        return state, self.exp_val(state)
    
class PQC_2A(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0 = self.AfRot()
        self.fR1 = self.AfRot()
        self.fR2 = self.AfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        Ry_data = Ry_layer(self.n_blocks, self.n_qubits, weights=x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)
        state = self.fR0(state)
        state = self.cnot(state)
        state = Ry_data(state)
        state = self.fR1(state)
        state = self.Entangle(state)
        state = Ry_data(state)
        state = self.fR2(state)
                
        return state, self.exp_val(state)
    
class PQC_2Y(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0 = self.YfRot()
        self.fR1 = self.YfRot()
        self.fR2 = self.YfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        Ry_data = Ry_layer(self.n_blocks, self.n_qubits, weights=x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data(state)
        state = self.fR0(state)
        state = self.cnot(state)
        state = Ry_data(state)
        state = self.fR1(state)
        state = self.Entangle(state)
        state = Ry_data(state)
        state = self.fR2(state)
                
        return state, self.exp_val(state)
    
class PQC_3D(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        self.fR0w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        self.fR1w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        self.fR2 = self.YfRot()

        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        
        Ry_data0 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR0w*x)
        Ry_data1 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR1w*x)
        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data0(state)
        state = self.cnot(state)
        state = Ry_data1(state)
        state = self.Entangle(state)
        state = self.fR2(state)
                
        return state, self.exp_val(state)
    
class PQC_3B(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: float = 1, grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        self.fR1 = self.YfRot()
        self.fR2w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        self.fR3 = self.YfRot()
        self.fR4w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        self.fR5 = self.YfRot()
        # if grant_init:
        #     copy_weights(self.fR2, self.fR3)
        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,0],[1,0]], [[1,4],[0,4]], [[0,1],[1,3]], [[1,2],[0,2]]],
                                        self.n_qubits)
        # self.Entangle = Entangle_layer([[[0,2],[1,2]],[[0,0],[2,2]],[[0,3],[3,3]],
        #                                 [[0,2],[5,2]],[[5,1],[1,3]],# [[8,0],[4,3]],
        #                                 ], self.n_qubits)
        

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]

        Ry_data0 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR0w*x)
        Ry_data2 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR2w*x)
        Ry_data4 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR4w*x)

        
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        #state[:, :, :, 0, 0] = 1
        
        #state = self.H(state) #Implicitly included in state
        
        state = Ry_data0(state)
        state = self.fR1(state)
        state = self.cnot(state)
        state = Ry_data2(state)
        state = self.fR3(state)
        state = self.Entangle(state)
        state = Ry_data4(state)
        state = self.fR5(state)
                
        return state, self.exp_val(state)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.shape[1] >= self.input_dim:
            x = torch.narrow(x, 1, 0, self.input_dim)
        x = self.flatten(x).float()
        logits = self.linear_relu_stack(x)
        return 0, logits

