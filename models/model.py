# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:45:01 2021

@author: Marius
"""

from .layers import Rx_layer, Ry_layer, Rz_layer, Hadamard_layer, CNOT_layer, Entangle_layer, krons
from .base_model import BaseModel
import torch 
import torch.nn as nn
import numpy as np
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

    
class TestModel(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                       
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
                
        def copy_weights(control_seq, target_seq):
            """Copies weights of one nn.Sequential layer into a target nn.Sequential layer and multiplies by a factor of -1"""
            for u in range(len(control_seq)):
                target_seq[u].weights = nn.Parameter(-control_seq[u].weights)
        
        self.cnot = self.cnot_(0,1)
        self.fR0w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR0w, 1,1)
        self.fR1w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR1w, 1,1)
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
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
       
        self.cnot = self.cnot_(0,1)
        # if grant_init:
        #     copy_weights(self.fR0, self.fR1)
        self.fR0w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR0w, 1,1)
        self.fR1 = self.YfRot()
        self.fR2w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR2w, 1,1)
        self.fR3 = self.YfRot()
        self.fR4w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR4w, 1,1)
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

class PQC_3E(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)   
       
        self.cnot = self.cnot_(0,1)

        self.fR0w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR0w, 1,1)
        self.fR1 = self.YfRot()
        self.fR2w = nn.Parameter(self.YfRot()[0].weights.view(1,-1))
        nn.init.uniform_(self.fR2w, 1,1)
        self.fR3 = self.YfRot()
        self.fR4 = self.YfRot()

        #self.H = Hadamard_layer(n_blocks, n_qubits)

        self.Entangle = Entangle_layer([[[0,-1],[1,0]]],
                                        self.n_qubits)

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)#[:,self.randperm]

        Ry_data0 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR0w*x)
        Ry_data2 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR2w*x)

        
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
        state = self.cnot(state)
        state = self.fR4(state)
        
                
        return state, self.exp_val(state)

class PQC_3V(PQC_3E):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)
        if "index" in kwargs.keys():
            self.idx = kwargs["index"]
        self.Entangle = Entangle_layer([], self.n_qubits) #No Entanglement between blocks
        self.cnot = Entangle_layer([], self.n_qubits) #No cnots within blocks
        self.single_qubit_Z(-1)
        Observable_ = self.Observable.view(1,1,1,-1,1).repeat(1,n_blocks,1,1,1)
        Observable_[0,:-1,0] = torch.ones_like(Observable_, dtype=torch.cdouble)[0,:-1,0]
        self.Observable = Observable_ #Output is just Z on final qubit in block 2
        
    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
        #x = x[:,self.idx].view(-1,1)
        Ry_data0 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR0w*x)
        Ry_data2 = Ry_layer(self.n_blocks, self.n_qubits, weights=self.fR2w*x)

        
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
        state = self.cnot(state)
        state = self.fR4(state)
        
                
        return state, self.exp_val(state)

class PQC_3W(PQC_3E):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)
        self.Entangle = Entangle_layer([], self.n_qubits) #No Entanglement between blocks
        self.single_qubit_Z(-1)
        Observable_ = self.Observable.view(1,1,-1,1).repeat(n_blocks,1,1,1)
        Observable_[:-1,0] = torch.ones(n_blocks-1, *self.Observable.shape)
        self.Observable = Observable_ #Output is just Z on final qubit in block 2
    
class PQC_3X(PQC_3E):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)
        self.Entangle = Entangle_layer([], self.n_qubits) #No Entanglement between blocks
        self.single_qubit_Z(-1) #Output is Z on final qubit in both blocks
        
class PQC_3Y(PQC_3E):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)
        #Product Z obeservable
        self.Entangle = Entangle_layer([], self.n_qubits) #No Entanglement between blocks
        self.cnot = Entangle_layer([], self.n_qubits) #No Entanglement in blocks

class PQC_3Z(PQC_3E):
    def __init__(self, n_blocks: int, n_qubits: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)
        self.Entangle = Entangle_layer([[[0,-1],[1,0]]], self.n_qubits) #Single Entanglement between blocks
        self.single_qubit_Z(-1)
        Observable_ = self.Observable.view(1,1,-1,1).repeat(n_blocks,1,1,1)
        Observable_[:-1,0] = torch.ones(n_blocks-1, *self.Observable.shape)
        self.Observable = Observable_ #Output is just Z on final qubit in block 2
        
        
class PQC_4A(BaseModel):
    def __init__(self, n_blocks: int, n_qubits: int, n_layers: int, weights_spread: list = [-np.pi/2,np.pi/2], grant_init: bool = False, **kwargs):
        super().__init__(n_blocks, n_qubits, weights_spread, grant_init)  
        self.n_layers = n_layers
        self.var = nn.Sequential(*[self.YfRot() for _ in range(n_layers)])
        self.dru = [nn.Parameter(self.YfRot(weights_spread=[1,1])[0].weights.view(1,-1)) for _ in range(n_layers)]
        self.cnot = self.cnot_(0,1)
        self.fvar = self.AfRot()
        self.Entangle = Entangle_layer([[[0,-1],[1,0]]], self.n_qubits) #Single Entanglement between blocks

    def forward(self, x):
        batch_size = x.shape[0]
        if x.shape[1] != self.n_blocks*self.n_qubits:
            x = torch.narrow(x, 1, 0, self.n_blocks*self.n_qubits)[:,self.randperm]
            
        state = torch.zeros((batch_size, self.n_blocks, 1, 2**self.n_qubits, 1), dtype=torch.cdouble)
        #state[:, :, :, :, 0] = 2**(-self.n_qubits/2)
        state[:, :, :, 0, 0] = 1
        
        for l in range(self.n_layers):
            Rx_data = Rx_layer(self.n_blocks, self.n_qubits, weights=self.dru[l]*x)
            state = Rx_data(state)
            state = self.var[l](state)
            state = self.cnot(state)
            
            if l == self.n_layers//2:
                state = self.Entangle(state)
            
        state = self.fvar(state)
        
        return state, self.exp_val(state)
        

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim
        self.dropout = nn.Dropout(0.25)
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 100)
        self.layer3 = nn.Linear(100, 50)
        self.layer4 = nn.Linear(50, 1)

    def forward(self, x):
        if x.shape[1] >= self.input_dim:
            x = torch.narrow(x, 1, 0, self.input_dim)
        x = self.flatten(x).float()
        out = nn.ReLU()(self.layer1(x))
        out = nn.ReLU()(self.layer2(out))
        out = self.dropout(out)
        out = nn.ReLU()(self.layer3(out))
        out = self.dropout(out)
        out = nn.Sigmoid()(self.layer4(out))
        return 0, out
    
class LinearNetwork(nn.Module):
    def __init__(self, input_dim, index):
        super(LinearNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.input_dim = input_dim
        self.layer1 = nn.Linear(1, 10)
        self.layer2 = nn.Linear(10, 1)
        self.idx = index

    def forward(self, x):
        if x.shape[1] >= self.input_dim:
            x = torch.narrow(x, 1, 0, self.input_dim)
        x = self.flatten(x).float()[:,self.idx].view(-1,1)
        out = self.layer1(x)
        out = self.layer2(out)
        return 0, out

